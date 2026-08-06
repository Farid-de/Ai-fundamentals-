import os
import time
import numpy as np
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Input, LSTM, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# seed رو ثابت می‌کنیم تا هر بار که کد رو اجرا می‌کنیم همون تقسیم‌بندی داده و
# همون نتیجه رو بگیریم (تکرارپذیر بشه).
np.random.seed(42)
tf.random.set_seed(42)
# تنظیمات کلی
# مهم: قبل از اجرا این سه مسیر رو با آدرس واقعی روی سیستم خودمون عوض می کنیم.
RAW_DIR = r"C:\Users\mm\Desktop\New folder\py\raw"                                        # پوشه‌ی فایل‌های خام .mat
FEATURE_CSV_PATH = r"C:\Users\mm\Desktop\New folder\py\feature_time_48k_2048_load_1.csv"  # فایل ویژگی‌های دستی
NPZ_PATH = r"C:\Users\mm\Desktop\New folder\py\CWRU_48k_load_1_CNN_data.npz"               # فایل داده‌ی آماده
OUTPUT_DIR = r"C:\Users\mm\Desktop\New folder\py\outputs"                                  # اینجا نتایج CSV ذخیره میشن

WINDOW_LEN = 1024     # هر پنجره‌ی سیگنال چند تا نمونه داشته باشه
STEP_LEN = 1024        # گام پنجره‌بندی؛ چون با WINDOW_LEN برابره یعنی پنجره‌ها روی هم نمی‌افتن
TEST_FRAC = 0.2        # چند درصد داده بره برای تست نهایی
VAL_FRAC = 0.1         # چند درصد از کل داده بره برای validation
CNN_EPOCHS = 200       # مدل‌های CNN چند بار کل داده رو ببینن
LSTM_EPOCHS = 40       # LSTM هر epoch‌ش کندتره، برای همین epoch کمتری بهش می‌دیم
LSTM_SEQ_LEN = 64      # قبل از دادن به LSTM، سیگنال رو به این طول کوتاه می‌کنیم
BATCH_SIZE = 32

# پارامترهای Data Augmentation مخصوص LSTM
LSTM_AUG_NOISE_STD = 0.02      # شدت نویز گاوسی
LSTM_AUG_SCALE_RANGE = (0.9, 1.1)  # بازه‌ی مقیاس‌دهی دامنه (یعنی ±۱۰٪)
LSTM_AUG_SHIFT_MAX = 5          # حداکثر گام جابه‌جایی زمانی
LSTM_AUG_N_AUGMENTS = 10        # به‌ازای هر نمونه، ۱۰ نسخه‌ی اضافه ساخته بشه

EARLY_STOPPING_PATIENCE = 15    # اگه val_loss این‌قدر epoch پشت‌سرهم بهتر نشد، آموزش رو متوقف کن

# اگه پوشه‌ی خروجی وجود نداشت، بسازش.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# این ۱۰ تا کلاس خطایی هستن که کل پروژه روشون کار می‌کنه. ترتیب این لیست مهمه،
# چون همه‌جای کد بر اساس همین ترتیب، اسم کلاس رو به عدد و برعکس تبدیل می‌کنه.
class_names = ["Ball_007", "Ball_014", "Ball_021",
               "IR_007", "IR_014", "IR_021",
               "OR_007", "OR_014", "OR_021", "Normal"]

# میگه هر فایل .mat خام مال کدوم نوع خطاست.
label_map = {
    'B007_1_123.mat': 'Ball_007', 'B014_1_190.mat': 'Ball_014', 'B021_1_227.mat': 'Ball_021',
    'IR007_1_110.mat': 'IR_007', 'IR014_1_175.mat': 'IR_014', 'IR021_1_214.mat': 'IR_021',
    'OR007_6_1_136.mat': 'OR_007', 'OR014_6_1_202.mat': 'OR_014', 'OR021_6_1_239.mat': 'OR_021',
    'Time_Normal_1_098.mat': 'Normal'
}
# هر اسم کلاس رو به یه عدد تبدیل می‌کنه مثلا {'Ball_007': 0, 'Ball_014': 1, ...}
label_to_idx = {name: idx for idx, name in enumerate(class_names)}

# توی فایل CSV، ستون 'fault' یه پسوند اضافه داره (مثل 'Ball_007_1' یا
# 'OR_007_6_1')، پس اول باید این‌ها رو به همون اسم استاندارد کلاس برگردونیم.
csv_fault_to_class = {
    'Ball_007_1': 'Ball_007', 'Ball_014_1': 'Ball_014', 'Ball_021_1': 'Ball_021',
    'IR_007_1': 'IR_007', 'IR_014_1': 'IR_014', 'IR_021_1': 'IR_021',
    'OR_007_6_1': 'OR_007', 'OR_014_6_1': 'OR_014', 'OR_021_6_1': 'OR_021',
    'Normal_1': 'Normal'
}
# این ۹ تا ستون همون ویژگی‌های آماری‌ان که برای آموزش SVM استفاده می‌کنیم.
CSV_FEATURE_COLS = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
# توابع کمکی - بارگذاری داده
def window_signal(sig, window_len, step_len):
    """یه سیگنال بلند رو تیکه‌تیکه می‌کنه، هر تیکه به طول ثابت window_len
    (وقتی step_len با window_len برابر باشه، تیکه‌ها روی هم نمی‌افتن).
    خروجی یه آرایه با شکل (n_windows, window_len) هست."""
    n_windows = (len(sig) - window_len) // step_len + 1
    w = np.zeros((n_windows, window_len))
    for i in range(n_windows):
        start_idx = i * step_len
        w[i, :] = sig[start_idx:start_idx + window_len]
    return w
def load_all_signals(raw_dir, label_map, window_len, step_len):
    """همه‌ی فایل‌های .mat خام رو یکی‌یکی باز می‌کنه، دو کانال DE و FE رو
    ازش درمیاره، تیکه‌تیکه‌شون می‌کنه، و در آخر همه‌ی فایل‌ها رو کنار هم
    می‌چینه و یه آرایه‌ی برچسب هم براشون می‌سازه."""
    de_list, fe_list, labels_list = [], [], []
    for fname, label in label_map.items():
        filepath = os.path.join(raw_dir, fname)
        if not os.path.exists(filepath):
            print(f"Warning: File {fname} not found.")
            continue
        mat = scipy.io.loadmat(filepath)
        keys = list(mat.keys())
        # اسم متغیرها توی هر فایل .mat یه‌کم فرق داره (مثلا 'X110_DE_time')،
        # پس به‌جای دنبال یه اسم ثابت گشتن، دنبال چیزی می‌گردیم که آخرش
        # 'DE_time' یا 'FE_time' باشه.
        de_key = next((k for k in keys if k.endswith('DE_time')), None)
        fe_key = next((k for k in keys if k.endswith('FE_time')), None)
        if not de_key or not fe_key:
            continue
        de = mat[de_key].flatten()
        fe = mat[fe_key].flatten()
        # بعضی فایل‌ها طول DE و FE‌شون یه‌کم فرق داره؛ برای اینکه نمونه‌به‌نمونه
        # با هم جفت بمونن، هر دو رو به کوتاه‌ترین طول برش می‌زنیم.
        n = min(len(de), len(fe))
        de, fe = de[:n], fe[:n]
        de_w = window_signal(de, window_len, step_len)
        fe_w = window_signal(fe, window_len, step_len)
        n_w = min(de_w.shape[0], fe_w.shape[0])
        de_list.append(de_w[:n_w])
        fe_list.append(fe_w[:n_w])
        labels_list.extend([label_to_idx[label]] * n_w)
        print(f"  {label:<10}: {n_w} windows")
    # پنجره‌های همه‌ی ۱۰ فایل رو کنار هم می‌چینیم تا یه آرایه‌ی بزرگ بشه.
    DE = np.vstack(de_list)
    FE = np.vstack(fe_list)
    labels = np.array(labels_list)
    return DE, FE, labels


def load_handcrafted_features(csv_path, feature_cols, fault_to_class, class_names):
    """فایل CSV ویژگی‌های آماری رو باز می‌کنه، برای آموزش SVM."""
    df = pd.read_csv(csv_path)
    X = df[feature_cols].values
    # برچسب‌های عجیب داخل CSV (مثل 'Ball_007_1') رو اول به اسم استاندارد کلاس
    # (مثل 'Ball_007') و بعد به عدد تبدیل می‌کنیم.
    mapped = df['fault'].map(fault_to_class)
    if mapped.isnull().any():
        unknown = df.loc[mapped.isnull(), 'fault'].unique()
        raise ValueError(f"Unrecognized fault labels in CSV: {unknown}")
    y = mapped.map({c: i for i, c in enumerate(class_names)}).values
    return X, y


def load_precomputed_npz(npz_path, class_names):
    """فایل npz آماده رو باز می‌کنه و هر نمونه‌ی ۳۲×۳۲ رو صاف می‌کنه تا بشه
    یه دنباله‌ی تک‌کاناله به طول ۱۰۲۴."""
    d = np.load(npz_path, allow_pickle=True)
    data = d['data']            # (n, 32, 32)
    labels_raw = d['labels']    # (n,) رشته‌ای؛ همین الانش هم با class_names جور درمیاد
    X_flat = data.reshape(data.shape[0], -1)  # (n, 1024), تک‌کاناله
    y = np.array([class_names.index(l) for l in labels_raw])
    return X_flat, y


def fft_features(window):
    """طیف دامنه‌ی FFT یه پنجره رو حساب می‌کنه و فقط نیمه‌ی اولش رو نگه
    می‌داره (نیمه‌ی دوم چون سیگنال حقیقیه، فقط تکرار آینه‌ایه، چیز جدیدی
    اضافه نمی‌کنه)."""
    spec = np.abs(np.fft.fft(window))
    return spec[:len(window) // 2]


def compute_min_max(X):
    """min و max رو محاسبه می‌کنه — این تابع باید فقط روی داده‌ی train
    صدا زده بشه. چرا مهمه؟ چون اگه min/max رو از کل داده (شامل
    validation/test) حساب کنیم و بعد همون رو برای نرمال‌سازی train هم
    استفاده کنیم، عملاً یه‌جور نشتی داده (data leakage) اتفاق افتاده:
    مدل قبل از آموزش، غیرمستقیم یه سرنخ از بازه‌ی داده‌ی تست دیده. برای
    همین دقیقاً مثل StandardScaler که برای SVM فقط روی train فیت می‌شه،
    اینجا هم min/max رو فقط از train یاد می‌گیریم."""
    return float(np.min(X)), float(np.max(X))


def apply_min_max(X, min_v, max_v):
    """X رو با یه min/max از پیش‌محاسبه‌شده (که باید از compute_min_max
    روی train اومده باشه) به بازه‌ی [0, 1] می‌بره. همون 1e-8 هم فقط برای
    اینه که اگه min و max یکی بودن، تقسیم بر صفر پیش نیاد."""
    return (X - min_v) / (max_v - min_v + 1e-8)


def get_class_weights(y):
    """وزن هر کلاس رو طوری حساب می‌کنه که کلاس‌های کم‌تکرار (مثلاً اگه
    فایل Normal کوتاه‌تر از فایل‌های خطا باشه و پنجره‌ی کمتری تولید کنه)
    توی loss وزن بیشتری بگیرن. این از تمایل مدل به کلاس‌های پرتکرارتر
    جلوگیری می‌کنه. خروجی یه دیکشنری {class_index: weight} هست که مستقیم
    به آرگومان class_weight توی model.fit داده می‌شه."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def downsample_for_lstm(X, target_len):
    """قبل از اینکه سیگنال رو به LSTM بدیم، طولش رو با max-pooling کوتاه
    می‌کنیم. چرا؟ چون LSTM با دنباله‌های خیلی طولانی اذیت می‌شه (گرادیانش
    محو می‌شه و چیزی یاد نمی‌گیره). از max گرفتن به‌جای میانگین استفاده
    کردیم چون می‌خوایم قله‌های سیگنال (که نشونه‌ی خطان) گم نشن.
    X شکلش (n, feat_len, channels) هست."""
    n, feat_len, ch = X.shape
    factor = feat_len // target_len
    trimmed_len = factor * target_len          # اگه چیزی اضافه بمونه که درست تقسیم نمی‌شه، حذفش می‌کنیم
    X_trim = X[:, :trimmed_len, :]
    # هر factor تا گام پشت‌سرهم رو یه گروه می‌کنیم و از هر گروه بیشترین مقدار رو برمی‌داریم.
    X_reshaped = X_trim.reshape(n, target_len, factor, ch)
    return X_reshaped.max(axis=2)


def augment_signal_batch(X, y, noise_std=0.02, scale_range=(0.9, 1.1),
                          shift_max=5, n_augments=2, seed=42):
    """
    Data Augmentation مخصوص دیتای LSTM.
    برای هر نمونه‌ی ورودی، n_augments نسخه‌ی افزوده‌شده (augmented) می‌سازه
    و به داده‌ی اصلی اضافه می‌کنه. سه نوع augmentation روی سیگنال اعمال می‌شه:

      1) Jitter (نویز گاوسی)
         یه نویز کوچیک و تصادفی به سیگنال اضافه می‌کنیم، شبیه نویز واقعی
         سنسور توی محیط صنعتی. این کار باعث می‌شه مدل به نویزهای جزئی
         حساس نباشه.

      2) Random Scaling (مقیاس‌دهی دامنه، ±۱۰٪)
         کل سیگنال هر نمونه رو با یک ضریب تصادفی بین scale_range (پیش‌فرض
         ۰.۹ تا ۱.۱، یعنی ±۱۰٪) ضرب می‌کنیم، شبیه‌سازیِ تغییرات دامنه‌ی
         ارتعاش که ممکنه به‌خاطر شرایط عملیاتی (بار، دما، سرعت) به وجود بیاد.

      3) Time Shift (جابه‌جایی زمانی دایره‌ای)
         سیگنال رو چند گام به جلو یا عقب می‌چرخونیم (np.roll). این کار
         باعث می‌شه مدل به محل دقیق شروع الگوی خطا وابسته نشه و روی خودِ
         شکل الگو تمرکز کنه، نه موقعیتش توی پنجره.
    """
    rng = np.random.default_rng(seed)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_augments):
        X_aug = X.copy()

        # 1) jitter: نویز گاوسی کوچک
        noise = rng.normal(0, noise_std, X_aug.shape)
        X_aug = X_aug + noise

        # 2) scaling: یک ضریب تصادفی جدا برای هر نمونه (نه هر گام زمانی)
        scale = rng.uniform(scale_range[0], scale_range[1], size=(X_aug.shape[0], 1, 1))
        X_aug = X_aug * scale

        # 3) time shift: جابه‌جایی دایره‌ای مستقل برای هر نمونه
        shifts = rng.integers(-shift_max, shift_max + 1, size=X_aug.shape[0])
        for i, s in enumerate(shifts):
            X_aug[i] = np.roll(X_aug[i], s, axis=0)

        # چون داده‌ی اصلی قبلاً با apply_min_max بین 0 و 1 نرمال شده،
        # بعد از اضافه‌کردن نویز و scale ممکنه مقداری از این بازه بیرون بزنه؛
        # دوباره clip می‌کنیم تا سازگار با بقیه‌ی پایپ‌لاین بمونه.
        X_aug = np.clip(X_aug, 0, 1)

        X_aug_list.append(X_aug)
        y_aug_list.append(y.copy())

    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)

    # داده‌های augmented رو با داده‌ی اصلی قاطی می‌کنیم (shuffle) تا توی
    # batch‌ها پشت‌سرهم قرار نگیرن؛ وگرنه هر batch ممکنه فقط از یک نوع
    # (اصلی یا augmented) پر بشه و آموزش رو مغرضانه کنه.
    perm = rng.permutation(len(X_final))
    return X_final[perm], y_final[perm]


# توابع کمکی - ساخت مدل‌ها
def build_cnn1d(feat_len, num_classes, num_channels=2):
    """مدل 1D-CNN: دو تا بلوک Conv1D، بعدش یه global pooling، و در آخر
    یه سر دسته‌بندی (dense). با پارامتر num_channels می‌تونیم هم نسخه‌ی
    ۲کاناله (DE+FE) و هم نسخه‌ی ۱کاناله (داده‌ی npz) رو با همین یه تابع بسازیم."""
    model = Sequential([
        Input(shape=(feat_len, num_channels)),
        Conv1D(32, 3, padding='same', activation='relu', name='conv1'),
        MaxPooling1D(2, strides=2, name='pool1'),
        Conv1D(64, 3, padding='same', activation='relu', name='conv2'),
        GlobalAveragePooling1D(name='gap'),   # کل محور زمان رو جمع می‌کنه، پس طول ورودی مهم نیست
        Dense(64, activation='relu', name='fc1'),
        Dropout(0.3, name='drop1'),
        Dense(num_classes, activation='softmax', name='output')  # احتمال هر کلاس خطا
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_lstm(feat_len, num_classes, num_channels=2):
    """مدل LSTM. لایه‌های BatchNormalization و گرفتن جلوی گرادیان
    (clipnorm) کمک می‌کنن آموزش پایدار بمونه؛ بدون این‌ها ممکنه LSTM
    فقط یه کلاس ثابت رو همیشه پیش‌بینی کنه (دقتش گیر کنه روی حدس تصادفی)."""
    model = Sequential([
        Input(shape=(feat_len, num_channels)),
        BatchNormalization(name='bn_in'),
        LSTM(64, return_sequences=False, name='lstm1'),
        BatchNormalization(name='bn_1'),
        Dropout(0.3, name='drop1'),
        Dense(64, activation='relu', name='fc1'),
        BatchNormalization(name='bn_2'),
        Dropout(0.3, name='drop2'),
        Dense(num_classes, activation='softmax', name='output')
    ])
    # clipnorm جلوی گرادیان‌های خیلی بزرگ رو می‌گیره، همون چیزی که باعث
    # می‌شد loss بره روی NaN و آموزش خراب بشه.
    opt = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_classifier(y_true, y_pred, name, class_names):
    """Accuracy، Precision/Recall/F1 و ماتریس درهم‌ریختگی رو حساب و چاپ
    می‌کنه، و یه ردیف برای جدول مقایسه‌ی نهایی برمی‌گردونه."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy (%): {acc * 100:.2f}%")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    return {"Model": name, "Accuracy": acc * 100, "Precision": prec, "Recall": rec, "F1-Score": f1}


def safe_to_csv(df, path, **kwargs):
    """مثل df.to_csv عمل می‌کنه، با این تفاوت که اگه فایل مقصد قفل یا
    غیرقابل‌نوشتن بود (مثلاً چون توی Excel یا یه ویوئر دیگه باز مونده،
    یا پوشه مجوز Write نداره)، به‌جای کرش‌کردن کل اسکریپت، با یه اسم
    جایگزین (همراه timestamp) ذخیره می‌کنه و یه هشدار چاپ می‌کنه.
    مسیر نهایی‌ای که واقعاً توش ذخیره شده رو برمی‌گردونه."""
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt_path = f"{base}_{int(time.time())}{ext}"
        print(f"Warning: '{path}' قابل نوشتن نبود (احتمالاً یه‌جا باز مونده). "
              f"به‌جاش ذخیره شد در:\n  {alt_path}")
        df.to_csv(alt_path, **kwargs)
        return alt_path


# فاز ۱: بارگذاری سیگنال‌های خام
print("=" * 60)
print("Ph1: loading raw vibration signals (DE + FE)")
print("=" * 60)
DE, FE, labels = load_all_signals(RAW_DIR, label_map, WINDOW_LEN, STEP_LEN)
print(f"\nTotal windows: {DE.shape[0]} | Window length: {WINDOW_LEN} samples")

# یه شاخص «شدت خطا» برای هر پنجره می‌سازیم: میانگین انرژی RMS دو کانال
# DE و FE. این رو بعداً به‌عنوان ورودی دوم سیستم فازی استفاده می‌کنیم
# (پایین‌تر توی فاز ۵ توضیح دادم چرا).
severity_raw = (np.sqrt(np.mean(DE ** 2, axis=1)) + np.sqrt(np.mean(FE ** 2, axis=1))) / 2


# فاز ۲: پیش‌پردازش FFT
print("\n" + "=" * 60)
print("P2: FFT preprocessing")
print("=" * 60)

num_samples = DE.shape[0]
feat_len = WINDOW_LEN // 2   # طول طیف FFT، یعنی نصف طول پنجره (چون fft_features فقط نیمه‌ی اول رو نگه می‌داره)
X_fft = np.zeros((num_samples, feat_len, 2))
for i in range(num_samples):
    X_fft[i, :, 0] = fft_features(DE[i])
    X_fft[i, :, 1] = fft_features(FE[i])
print(f"  FFT feature shape (before scaling): {X_fft.shape}")

# داده رو ۸۰/۲۰ برای train و test تقسیم می‌کنیم، طوری که نسبت هر کلاس توی هر
# دو تا حفظ بشه (stratify). severity_raw رو هم همراه بقیه تقسیم می‌کنیم فقط
# تا sev_test با y_test/X_test هم‌ردیف بمونه.
# نکته‌ی مهم: تقسیم رو *قبل* از نرمال‌سازی انجام می‌دیم، نه بعدش! اگه
# اول کل X_fft رو نرمال می‌کردیم و بعد تقسیم می‌کردیم، min/max از داده‌ی
# تست هم توی نرمال‌سازیِ داده‌ی train نشت می‌کرد (data leakage) و دقتی که
# روی تست می‌گرفتیم واقعی نبود.
X_temp, X_test, y_temp, y_test, sev_temp, sev_test = train_test_split(
    X_fft, labels, severity_raw, test_size=TEST_FRAC, stratify=labels, random_state=42
)
# از همون ۸۰٪ باقی‌مونده، یه تیکه رو برای validation جدا می‌کنیم. val_adj
# نسبت رو طوری حساب می‌کنه که در نهایت validation دقیقا VAL_FRAC از کل
# داده‌ی اصلی بشه (مثلا ۰.۱ تقسیم بر ۰.۸ = ۰.۱۲۵ از همون ۸۰٪).
val_adj = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=42
)

# min/max رو فقط از X_train یاد می‌گیریم (دقیقاً مثل StandardScaler که
# برای SVM فقط روی train فیت شد)، و همون دو عدد رو برای نرمال‌سازی
# train/val/test استفاده می‌کنیم.
fft_min, fft_max = compute_min_max(X_train)
X_train = apply_min_max(X_train, fft_min, fft_max)
X_val = apply_min_max(X_val, fft_min, fft_max)
X_test = apply_min_max(X_test, fft_min, fft_max)
print(f"  Train/Val/Test sizes: {X_train.shape[0]} / {X_val.shape[0]} / {X_test.shape[0]}")

# هر مدل که آموزشش تموم بشه، نتیجه‌ش رو اینجا اضافه می‌کنیم در آخر همه‌شون
# رو با هم توی یه جدول مقایسه‌ای می‌ذاریم.
comparison_rows = []

# فاز ۳a: SVM (مدل پایه)، آموزش‌دیده روی ویژگی‌های آماری CSV

print("\n" + "=" * 60)
print("P3a: SVM")
print("=" * 60)

X_csv, y_csv = load_handcrafted_features(FEATURE_CSV_PATH, CSV_FEATURE_COLS, csv_fault_to_class, class_names)
print(f"  handcrafted feature matrix shape: {X_csv.shape}")

# این CSV تعداد ردیفش با پایپ‌لاین سیگنال خام فرق داره، پس نمی‌تونیم از همون
# تقسیم بالا استفاده کنیم - یه تقسیم train/test جدا و مستقل براش می‌گیریم.
X_csv_train, X_csv_test, y_csv_train, y_csv_test = train_test_split(
    X_csv, y_csv, test_size=TEST_FRAC, stratify=y_csv, random_state=42
)

# SVM به مقیاس ویژگی‌ها حساسه، پس اول استانداردشون می‌کنیم (میانگین صفر،
# واریانس یک). این استانداردسازی رو فقط روی داده‌ی train یاد می‌گیریم و
# همون رو روی test اعمال می‌کنیم (هیچ‌وقت scaler رو روی test فیت نمی‌کنیم،
# وگرنه data leakage میشه).
svm_scaler = StandardScaler()
X_csv_train_s = svm_scaler.fit_transform(X_csv_train)
X_csv_test_s = svm_scaler.transform(X_csv_test)

svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_csv_train_s, y_csv_train)
svm_pred = svm_model.predict(X_csv_test_s)

comparison_rows.append(evaluate_classifier(y_csv_test, svm_pred, "SVM (baseline)", class_names))

# فاز ۳b: 1D-CNN (پایپ‌لاین اصلی: سیگنال خام -> FFT -> CNN)

print("\n" + "=" * 60)
print("Phase 3b: 1D-CNN (raw signal + FFT)")
print("=" * 60)

cnn_model = build_cnn1d(feat_len, len(class_names), num_channels=2)
cnn_model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=CNN_EPOCHS, batch_size=BATCH_SIZE, verbose=1,
    class_weight=get_class_weights(y_train),
    callbacks=[EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True)]
)
cnn_probs = cnn_model.predict(X_test)     # احتمال هر کلاس، شکلش (n_test, num_classes)
cnn_pred = np.argmax(cnn_probs, axis=1)   # کلاسی که بیشترین احتمال رو داره، همون پیش‌بینی نهاییه

comparison_rows.append(evaluate_classifier(y_test, cnn_pred, "1D-CNN", class_names))

# فاز ۳c: LSTM (فقط سیگنال FFT) + Data Augmentation

print("\n" + "=" * 60)
print("P3c: LSTM (FFT) + Augmentation")
print("=" * 60)
print(f"Sequence length after downsampling: {LSTM_SEQ_LEN}")

# طیف FFT رو به همون طول LSTM_SEQ_LEN کوتاه می‌کنیم (max-pooling روی
# محور فرکانس/طول) تا با LSTM سازگار بشه. اینجا دیگه سیگنال خام رو به
# ورودی LSTM اضافه نمی‌کنیم؛ ورودی مدل فقط ۲ کاناله‌ست: [DE_fft, FE_fft].
X_train_lstm = downsample_for_lstm(X_train, LSTM_SEQ_LEN)
X_val_lstm = downsample_for_lstm(X_val, LSTM_SEQ_LEN)
X_test_lstm = downsample_for_lstm(X_test, LSTM_SEQ_LEN)


# Data Augmentation 
# فقط داده‌ی train رو augment می‌کنیم؛ val و test دست‌نخورده می‌مونن تا
# ارزیابی مدل واقعی و بدون leakage باشه.
X_train_lstm_aug, y_train_aug = augment_signal_batch(
    X_train_lstm, y_train,
    noise_std=LSTM_AUG_NOISE_STD,
    scale_range=LSTM_AUG_SCALE_RANGE,
    shift_max=LSTM_AUG_SHIFT_MAX,
    n_augments=LSTM_AUG_N_AUGMENTS,
    seed=42
)
print(f"  LSTM training size after augmentation:  {X_train_lstm_aug.shape[0]}")

lstm_model = build_lstm(LSTM_SEQ_LEN, len(class_names), num_channels=X_train_lstm.shape[-1])
lstm_model.fit(
    X_train_lstm_aug, y_train_aug, validation_data=(X_val_lstm, y_val),
    epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, verbose=1,
    class_weight=get_class_weights(y_train_aug),
    callbacks=[EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True)]
)
lstm_probs = lstm_model.predict(X_test_lstm)
lstm_pred = np.argmax(lstm_probs, axis=1)

comparison_rows.append(evaluate_classifier(y_test, lstm_pred, "LSTM (Augmented)", class_names))

# فاز ۳d: یه ردیف اضافه - 1D-CNN که روی داده‌ی آماده‌ی npz آموزش دیده
# اینجا از فایل CWRU_48k_load_1_CNN_data.npz به‌عنوان یه منبع داده‌ی کاملاً
# جدا استفاده می‌کنیم، مستقل از پایپ‌لاین پنجره‌بندی/FFT خودمون.

print("\n" + "=" * 60)
print("P3d: 1D-CNN on precomputed windowed data (npz, single channel)")
print("=" * 60)

X_npz, y_npz = load_precomputed_npz(NPZ_PATH, class_names)
print(f"  Precomputed data shape (before scaling): {X_npz.shape}")

# این فایل npz تعداد نمونه‌ش (۴۶۰۰) با پایپ‌لاین سیگنال خام (حدود ۴۶۴۰) فرق
# داره، پس این هم یه تقسیم train/test جدا و مستقل خودش رو می‌گیره.
# نکته: تقسیم رو *قبل* از نرمال‌سازی انجام می‌دیم (دقیقاً مثل پایپ‌لاین
# اصلی)، وگرنه min/max از val/test هم توی نرمال‌سازیِ train نشت می‌کرد.
X_npz_temp, X_npz_test, y_npz_temp, y_npz_test = train_test_split(
    X_npz, y_npz, test_size=TEST_FRAC, stratify=y_npz, random_state=42
)
X_npz_train, X_npz_val, y_npz_train, y_npz_val = train_test_split(
    X_npz_temp, y_npz_temp, test_size=val_adj, stratify=y_npz_temp, random_state=42
)

# min/max رو فقط از X_npz_train یاد می‌گیریم و همون رو روی val/test اعمال
# می‌کنیم (همون منطق compute_min_max/apply_min_max که برای FFT هم استفاده شد).
npz_min, npz_max = compute_min_max(X_npz_train)
X_npz_train = apply_min_max(X_npz_train, npz_min, npz_max)
X_npz_val = apply_min_max(X_npz_val, npz_min, npz_max)
X_npz_test = apply_min_max(X_npz_test, npz_min, npz_max)

# Conv1D حتما یه محور کانال می‌خواد، پس (n, 1024) رو به (n, 1024, 1)
# تبدیل می‌کنیم چون این داده تک‌کاناله‌ست.
X_npz_train_r = X_npz_train[..., np.newaxis]
X_npz_val_r = X_npz_val[..., np.newaxis]
X_npz_test_r = X_npz_test[..., np.newaxis]

npz_cnn_model = build_cnn1d(X_npz.shape[1], len(class_names), num_channels=1)
npz_cnn_model.fit(
    X_npz_train_r, y_npz_train, validation_data=(X_npz_val_r, y_npz_val),
    epochs=CNN_EPOCHS, batch_size=BATCH_SIZE, verbose=1,
    class_weight=get_class_weights(y_npz_train),
    callbacks=[EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE,
                              restore_best_weights=True)]
)
npz_cnn_probs = npz_cnn_model.predict(X_npz_test_r)
npz_cnn_pred = np.argmax(npz_cnn_probs, axis=1)

comparison_rows.append(evaluate_classifier(y_npz_test, npz_cnn_pred, "1D-CNN (Precomputed Data)", class_names))

# فاز ۳e: ردیف "CNN + Fuzzy System" برای جدول مقایسه‌ای
# (این عددها همون عددهای 1D-CNN اصلی‌ان - چرا؟ توضیحش بالای فایل هست)

cnn_fuzzy_row = evaluate_classifier(y_test, cnn_pred, "CNN + Fuzzy System", class_names)
comparison_rows.append(cnn_fuzzy_row)


# جدول مقایسه‌ی نهایی

comparison_df = pd.DataFrame(comparison_rows)[["Model", "Accuracy", "Precision", "Recall", "F1-Score"]]
comparison_df = comparison_df.rename(columns={"Accuracy": "Accuracy (%)"})

print("\n" + "=" * 60)
print(" Comparison of different models")
print("=" * 60)
print(comparison_df.to_string(index=False))

comparison_csv_path = os.path.join(OUTPUT_DIR, "model_comparison_table.csv")
comparison_csv_path = safe_to_csv(comparison_df, comparison_csv_path, index=False, encoding="utf-8-sig")
print(f"\ntable saved to:\n  {comparison_csv_path}")

# فاز ۵: سیستم فازی (Mamdani) روی کل مجموعه‌ی تست

print("\n" + "=" * 60)
print("P5: Fuzzy Inference System (Mamdani) on the FULL test set")
print("=" * 60)
# این بخش دقیقا همون جور منطق فازی که توی معماری پیشنهادی خواسته شده رو
# پیاده می‌کنه: یه سیستم Mamdani با دو ورودی فازی‌شده، یه دسته قانون
# IF-AND-THEN، و یه امتیاز ریسک در آخر.
#   ورودی ۱: prob_fault  -> همون اطمینان CNN، یعنی احتمالی که به کلاس پیش‌بینی‌شده داده
#   ورودی ۲: severity    -> یه شاخص کلی برای «چقدر وضعیت بده». توی معماری
#                           پیشنهادی می‌تونست نرخ تغییرات دما یا جریان باشه؛
#                           ولی چون این دیتاست فقط سیگنال ارتعاش داره، از
#                           انرژی RMS همون سیگنال استفاده کردیم. خودِ منطق
#                           (فازی‌سازی -> قوانین -> نتیجه‌گیری) فرقی نمی‌کنه
#                           هرچی که این ورودی دوم باشه.
# ساختار قانون (همون فرمتی که خواسته شده بود):
#   IF (prob_fault بالاست) AND (severity بالاست) THEN (ریسک = بحرانی)

#  تعریف متغیرهای فازی (بازه‌ی هر کدوم)
prob_fault = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'prob_fault')
severity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'severity')
risk_level = ctrl.Consequent(np.arange(0, 101, 1), 'risk_level')

#  توابع عضویت (مثلثی: trimf(بازه, [چپ, قله, راست]))
prob_fault['low'] = fuzz.trimf(prob_fault.universe, [0, 0, 0.5])
prob_fault['high'] = fuzz.trimf(prob_fault.universe, [0.4, 1, 1])

severity['normal'] = fuzz.trimf(severity.universe, [0, 0, 0.6])
severity['high'] = fuzz.trimf(severity.universe, [0.4, 1, 1])

risk_level['safe'] = fuzz.trimf(risk_level.universe, [0, 0, 50])
risk_level['warning'] = fuzz.trimf(risk_level.universe, [30, 50, 80])
risk_level['critical'] = fuzz.trimf(risk_level.universe, [60, 100, 100])

# قوانین (IF-AND-THEN)
rule1 = ctrl.Rule(prob_fault['high'] & severity['high'], risk_level['critical'])
rule2 = ctrl.Rule(prob_fault['low'] & severity['normal'], risk_level['safe'])
rule3 = ctrl.Rule(prob_fault['high'] & severity['normal'], risk_level['warning'])
rule4 = ctrl.Rule(prob_fault['low'] & severity['high'], risk_level['warning'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# مقدار severity (که از RMS اومده) رو با کمترین/بیشترین مقدارش توی مجموعه‌ی
# تست به بازه‌ی [0, 1] می‌بریم تا با بازه‌ی سیستم فازی جور دربیاد.
# مقدار severity (که از RMS اومده) رو با min/max یاد می‌گیریم - اما این بار
# از sev_temp (یعنی همون قسمت train+val، نه خودِ تست) چون سیستم فازی هم
# باید مثل بقیه‌ی مدل‌ها فقط با آماری که از قبل در دسترسه کالیبره بشه؛
# اگه از min/max خودِ تست استفاده می‌کردیم، یعنی موقع inference واقعی به
# چیزی نیاز داشتیم که هنوز نمی‌دونیمش.
sev_min, sev_max = np.min(sev_temp), np.max(sev_temp)
sev_test_norm = (sev_test - sev_min) / (sev_max - sev_min + 1e-8)
sev_test_norm = np.clip(sev_test_norm, 0, 1)
# برای هر نمونه‌ی تست، بیشترین احتمالی که CNN اصلی داده رو به‌عنوان ورودی
# "prob_fault" به سیستم فازی می‌دیم.
cnn_prob_best = np.max(cnn_probs, axis=1)

# حالا سیستم فازی رو یه‌بار به‌ازای هر نمونه‌ی تست اجرا می‌کنیم (یعنی
# واقعاً برای همه‌ی نمونه‌ها، نه فقط چندتاش) و امتیاز ریسک هر کدوم رو
# جمع می‌کنیم.
n_test = len(y_test)
risk_scores = np.zeros(n_test)
for i in range(n_test):
    risk_sim.input['prob_fault'] = float(np.clip(cnn_prob_best[i], 0, 1))
    risk_sim.input['severity'] = float(np.clip(sev_test_norm[i], 0, 1))
    risk_sim.compute()
    risk_scores[i] = risk_sim.output['risk_level']

# امتیاز پیوسته‌ی ۰ تا ۱۰۰ رو به سه دسته‌ی قابل‌فهم تبدیل می‌کنیم.
risk_categories = np.where(
    risk_scores <= 50, "Safe",
    np.where(risk_scores <= 80, "Warning", "Critical")
)

# برای هر نمونه‌ی تست یه ردیف می‌سازیم: برچسب واقعیش، اطمینان CNN، شدت،
# امتیاز ریسک فازی، و دسته‌ای که توش قرار می‌گیره.
results_df = pd.DataFrame({
    'TrueLabel': [class_names[idx] for idx in y_test],
    'CNN_Prob': cnn_prob_best,
    'Severity_Norm': sev_test_norm,
    'FuzzyRiskScore': risk_scores,
    'RiskCategory': risk_categories
})

print(f"\nRisk level distribution on the FULL test set ({n_test} samples) ")
print(results_df['RiskCategory'].value_counts())

# همه‌ی نمونه‌ها رو ذخیره می‌کنیم (نه فقط چندتاش) تا بتونیم هر پیش‌بینی رو
# جداگونه بررسی کنیم.
results_csv_path = os.path.join(OUTPUT_DIR, "fuzzy_risk_results_full.csv")
results_csv_path = safe_to_csv(results_df, results_csv_path, index=False, encoding="utf-8-sig")
print(f"\nFuzzy risk results for all {n_test} test samples saved to:\n  {results_csv_path}")

print("\n avg risk score by fault type (all  samples)")
mean_risk_by_class = results_df.groupby('TrueLabel')['FuzzyRiskScore'].mean().reset_index()
mean_risk_by_class = mean_risk_by_class.sort_values(by='FuzzyRiskScore', ascending=False)
print(mean_risk_by_class.to_string(index=False))