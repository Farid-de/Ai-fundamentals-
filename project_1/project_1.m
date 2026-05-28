rng(42)
%پیشپردازش
T = readtable('titanic.csv');%  بارگذاری دیتاست
T.FamilySize = T.SibSp + T.Parch + 1;
disp(' بررسی کلی ساختار داده و مقادیر گمشده ');% ۲. تحلیل توزیع کلاس‌ها و مقادیر گمشده
summary(T); % نمایش خلاصه وضعیت داده‌ها
disp(' توزیع کلاس Survived ');
tabulate(T.Survived); % تحلیل توزیع کلاس‌ها
%  حذف ستون‌های غیرضروری (Name, Ticket, Cabin)
T.Name = [];
T.Ticket = [];
T.Cabin = [];
%  مقداردهی به داده‌های گمشده Age با میانگین 
T.Age = fillmissing(T.Age, 'constant', mean(T.Age, 'omitnan'));
%  تبدیل فیچرهای categorical به عددی 
% الف) تبدیل Sex: Male=0, Female=1
T.Sex = double(strcmp(T.Sex, 'female')); 
% ب) تبدیل Embarked با One-hot encoding
T.Embarked = categorical(T.Embarked);
T.Embarked = fillmissing(T.Embarked, 'constant', mode(T.Embarked)); % مدیریت مقادیر گمشده
embarked_encoded = onehotencode(T(:, 'Embarked')); 
T = [T, embarked_encoded]; 
T.Embarked = []; % حذف ستون اصلی
% جدا کردن هدف  از ورودی‌ها
targets = T.Survived'; 
T.Survived = []; 
%  نرمال‌سازی فیچرها با zscore 
X_matrix = table2array(T); 
inputs = zscore(X_matrix)'; 
% طراحی mlp 
%  ساخت شبکه mlp برای طبقه‌بندی 
%  تعریف معماری: ۲ لایه مخفی [16 8] و تابع آموزش trainscg 
net = patternnet([16 8], 'trainscg');
%  تنظیم تابع فعال‌ساز (tansig برای لایه‌های مخفی و logsig برای خروجی)
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'logsig';

%  تنظیمات پارامترهای آموزش 
% الف) تقسیم داده‌ها: ۷۰٪ آموزش، ۱۵٪ اعتبارسنجی و ۱۵٪ تست
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;
% ب) تنظیم تعداد Epoch ها بر روی ۲۰۰
net.trainParam.epochs = 200;
% ج) تنظیم تابع کارایی بر روی crossentropy
net.performFcn = 'crossentropy';
% آموزش شبکه 
[net, tr] = train(net, inputs, targets);
%  آموزش و ارزیابی
%  گرفتن خروجی‌های شبکه 
outputs = net(inputs);
%  رسم منحنی‌های Performance 
figure; plotperform(tr); 
%  نمایش ماتریس اغتشاش (Confusion Matrix) 
figure; plotconfusion(targets, outputs);
% رسم نمودار ROC 
figure; plotroc(targets, outputs);
%  محاسبه معیارهای ارزیابی (Accuracy, Precision, Recall, F1) 
% استخراج شاخص‌های مربوط به داده‌های تست
testIdx = tr.testInd;
T_test = targets(testIdx);

%  اصلاح خطا: تبدیل خروجی منطقی به Double برای هماهنگی با T_test 
Y_test = double(outputs(testIdx) > 0.5); 
% محاسبه ماتریس اغتشاش برای داده‌های تست
[C, ~] = confusionmat(T_test, Y_test);
tp = C(2,2); tn = C(1,1); fp = C(1,2); fn = C(2,1);
% فرمول‌های ارزیابی
accuracy = (tp + tn) / sum(C(:));
precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1score = 2 * (precision * recall) / (precision + recall);
% نمایش نتایجه 
fprintf('\n نتایج نهایی ارزیابی روی داده‌های تست (Test Set) \n');
fprintf('دقت کلی (Accuracy): %.2f%%\n', accuracy * 100);
fprintf('صحت (Precision):    %.2f\n', precision);
fprintf('فراخوانی (Recall):   %.2f\n', recall);
fprintf('نمره ترکیبی (F1):    %.2f\n', f1score);
%تحلیل
errors_idx = find(Y_test ~= T_test);
fprintf('\nتحلیل خطاها \n');
fprintf('تعداد کل نمونه‌های اشتباه تشخیص داده شده در تست: %d مورد\n', length(errors_idx));
if ~isempty(errors_idx)
    fprintf('اندیس نمونه‌های خطا در کل دیتاست: \n');
    disp(testIdx(errors_idx(1:min(10, end)))); % نمایش ۱۰ مورد اول
end
