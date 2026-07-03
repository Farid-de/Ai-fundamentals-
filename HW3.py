# وارد کردن کتابخانه‌های مورد نیاز
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import pygad
# بارگذاری مجموعه داده سرطان سینه (Breast Cancer)
data = load_breast_cancer()
X = data.data
y = data.target
# استانداردسازی داده‌ها (تغییر مقیاس داده‌ها به طوری که میانگین صفر و واریانس یک شود)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# تقسیم داده‌ها به دو مجموعه آموزش (70 درصد) و آزمون (30 درصد)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# تعریف تابع فعال‌ساز سیگموئید (Sigmoid) برای شبکه عصبی
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# تابع پیش‌بینی که وزن‌های شبکه عصبی را از جواب الگوریتم ژنتیک (solution) استخراج کرده و خروجی را حساب می‌کند
def predict(solution, X_data):
    # استخراج وزن‌ها و بایاس‌های لایه اول (لایه پنهان با 5 نورون و 30 ورودی)
    W1 = solution[0:150].reshape(30, 5) # 30 ویژگی ورودی ضربدر 5 نورون
    b1 = solution[150:155].reshape(5)   # 5 بایاس برای لایه اول 
    # استخراج وزن‌ها و بایاس لایه دوم (لایه خروجی با 1 نورون)
    W2 = solution[155:160].reshape(5, 1) # 5 ورودی از لایه قبل ضربدر 1 نورون خروجی
    b2 = solution[160]                   # 1 بایاس برای خروجی
    # محاسبات انتشار رو به جلو (Forward Propagation) برای لایه اول
    Z1 = np.dot(X_data, W1) + b1
    A1 = sigmoid(Z1)
    # محاسبات انتشار رو به جلو برای لایه دوم (خروجی)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    # تبدیل احتمالات به کلاس‌های باینری 0 و 1 (آستانه 0.5)
    predictions = (A2 >= 0.5).astype(int).flatten()
    return predictions
# تابع برازندگی (Fitness Function) برای ارزیابی عملکرد هر کروموزوم (مدل) در الگوریتم ژنتیک
def fitness_func(ga_instance, solution, solution_idx):
    # پیش‌بینی روی داده‌های آموزش
    predictions = predict(solution, X_train)
    # محاسبه دقت (Accuracy) به عنوان مقدار برازندگی
    fitness = accuracy_score(y_train, predictions)
    return fitness
# لیست‌هایی برای ذخیره روند تغییرات بهترین و میانگین برازندگی در طول نسل‌ها
best_fitness_history = []
avg_fitness_history = []
# تابعی که در پایان هر نسل (Generation) توسط الگوریتم ژنتیک فراخوانی می‌شود
def on_generation(ga_instance):
    # ثبت بهترین برازندگی در نسل فعلی
    best_fitness_history.append(ga_instance.best_solution()[1])
    # محاسبه و ثبت میانگین برازندگی کل جمعیت در نسل فعلی
    avg_fitness = np.mean(ga_instance.last_generation_fitness)
    avg_fitness_history.append(avg_fitness)
# تنظیمات و ایجاد نمونه‌ی الگوریتم ژنتیک
ga_instance = pygad.GA(
    num_generations=100,                  # تعداد کل نسل‌ها
    num_parents_mating=int(50 * 0.8),     # تعداد والدینی که برای تولید مثل انتخاب می‌شوند (80٪ جمعیت)
    fitness_func=fitness_func,            # تابع ارزیابی برازندگی
    sol_per_pop=50,                       # تعداد کروموزوم‌ها (جواب‌ها) در هر جمعیت
    num_genes=161,                        # تعداد کل ژن‌ها (تعداد کل وزن‌ها و بایاس‌های شبکه عصبی: 150+5+5+1)
    parent_selection_type="tournament",   # روش انتخاب والدین (تورنمنت)
    K_tournament=3,                       # اندازه تورنمنت برای انتخاب والدین
    crossover_type="single_point",        # روش تقاطع یا ترکیب ژن‌ها (تک نقطه‌ای)
    crossover_probability=0.8,            # احتمال وقوع تقاطع (80 درصد)
    mutation_type="random",               # روش جهش ژنتیکی (تصادفی)
    mutation_probability=0.05,            # احتمال وقوع جهش (5 درصد)
    on_generation=on_generation           # تابعی که در پایان هر نسل اجرا می‌شود
)
# اجرای الگوریتم ژنتیک
ga_instance.run()
# استخراج بهترین جواب و مقدار برازندگی آن پس از پایان اجرای الگوریتم
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
# آزمایش بهترین مدل پیدا شده روی داده‌های آزمون (تست)
test_predictions = predict(best_solution, X_test)
# محاسبه معیارهای ارزیابی مدل روی داده‌های آزمون
cm = confusion_matrix(y_test, test_predictions)   # ماتریس درهم‌ریختگی
precision = precision_score(y_test, test_predictions) # معیار دقت (Precision)
recall = recall_score(y_test, test_predictions)       # معیار فراخوانی (Recall)
# چاپ نتایج
print("Best Fitness (Training Accuracy):", best_solution_fitness)
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
# رسم نمودار روند همگرایی الگوریتم ژنتیک
plt.plot(best_fitness_history, label="Best Fitness")
plt.plot(avg_fitness_history, label="Average Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Algorithm Convergence")
plt.legend()
plt.show()