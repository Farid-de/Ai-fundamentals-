% با تنظیم هسته تولید اعداد تصادفی (Seed)، نتایج در هر بار اجرا دقیقاً یکسان 
rng(42); 
filename = 'iris.csv'; 
try
    % خواندن دیتا
    raw_data = readmatrix(filename);
    X = raw_data(:, 1:4); % ستون‌های ۱ تا ۴ ویژگی‌های عددی گلبرگ و کاسبرگ هستند
    y_raw = raw_data(:, 5); % ستون ۵ برچسب کلاس‌ها است
    
    % اگر ستون ۵ به دلیل متنی بودن به NaN تبدیل شده باشد، وارد بخش catch می‌شویم
    if any(isnan(y_raw))
        error('برچسب‌ها متنی هستند؛ باید به دسته عددی تبدیل شوند.');
    end
    y = y_raw;
catch
 %خواندن در صورت متنی بودن
    % (مانند رشته‌های Iris-setosa, Iris-versicolor, Iris-virginica)
    dataTable = readtable(filename, 'ReadVariableNames', false);
    X = table2array(dataTable(:, 1:4)); % تبدیل ۴ ستون اول به ماتریس ویژگی‌ها
    y_labels = dataTable{:, 5};         % استخراج برچسب‌های متنی
    y = grp2idx(y_labels);              % تبدیل خودکار متن‌ها به شماره کلاس (1، 2 و 3)
end
% برای اینکه ویژگی‌هایی با مقیاس بزرگتر بر یادگیری KNN غالب نشوند، 
% میانگین هر ویژگی را صفر و انحراف معیار آن را برابر ۱ می‌کنیم.
% فرمول ریاضی: z = (x - mean) / std
X_normalized = (X - mean(X)) ./ std(X);
num_features = size(X_normalized, 2); % تعداد ویژگی‌ها (که برای دیتاست ایریس برابر ۴ است)
% برای ارزیابی پایدار و علمی، داده‌ها را یک‌بار در ابتدا به نسبت ۷۰ به ۳۰ تقسیم می‌کنیم.
cv = cvpartition(y, 'Holdout', 0.3);
X_train = X_normalized(training(cv), :); % داده‌های آموزش (۷۰٪)
y_train = y(training(cv));               % برچسب‌های آموزش
X_test  = X_normalized(test(cv), :);     % داده‌های تست (۳۰٪)
y_test  = y(test(cv));                   % برچسب‌های تست
%تنظیم پارامتر های الگوریتم ژنتیک
pop_size = 30;           % تعداد جمیعت
num_generations = 100;    % تعداد نسل ها
pc = 0.8;                %احتمال تقطیع 
pm = 0.01;                % نرخ جهش
tournament_size = 3;     % تعداد افراد شرکت‌کننده در هر رقابت تورنمنت برای انتخاب والدین
%ایجاد جمیعت اولیه
% ایجاد یک ماتریس باینری به ابعاد (pop_size * num_features). هر سطر یک کروموزوم است.
% مقدار 1 یعنی ویژگی انتخاب شده و مقدار 0 یعنی ویژگی انتخاب نشده است.
population = randi([0, 1], pop_size, num_features);
% اعمال گروموزوم تمام صفر
% کروموزوم [0,0,0,0] غیرمجاز است؛ چون حداقل یک ویژگی باید برای طبقه‌بندی انتخاب شود.
for i = 1:pop_size
    if sum(population(i, :)) == 0
        % اگر کروموزومی تماماً صفر بود، یکی از ژن‌های آن را به تصادف ۱ می‌کنیم.
        population(i, randi(num_features)) = 1; 
    end
end
%تعریف ارایه ها برای ثبت فیتنس میانگین و بهترین
best_fitness_history = zeros(num_generations, 1);
avg_fitness_history = zeros(num_generations, 1);
% متغیرهایی برای نگهداری بهترین پاسخ سراسری پیدا شده در طول کل نسل‌ها
global_best_fitness = -inf;
global_best_chromosome = [];
%بخش اصلی الگوریتم ژنتیک
for gen = 1:num_generations
    fitness_values = zeros(pop_size, 1); % آرایه ذخیره برازندگی افراد نسل فعلی
        % ارزیابی فیتنس
    for i = 1:pop_size
        fitness_values(i) = evaluate_fitness(population(i, :), X_train, y_train, X_test, y_test);
    end
        % یافتن بهترین فرد در نسل فعلی
    [current_best_fit, best_idx] = max(fitness_values);
        % به‌روزرسانی بهترین پاسخ سراسری 
    if current_best_fit > global_best_fitness
        global_best_fitness = current_best_fit;
        global_best_chromosome = population(best_idx, :);
    end
        % ذخیره اطلاعات نسل جاری جهت رسم نمودار همگرایی نهایی
    best_fitness_history(gen) = global_best_fitness;
    avg_fitness_history(gen) = mean(fitness_values);
    
    % چاپ وضعیت همگرایی نسل جاری در Command Window جهت مانیتورینگ و دیباگ (تصویر ۳)
    fprintf('Gen %d: Best Fitness = %.4f | Avg Fitness = %.4f | Selected Features = %d\n', ...
        gen, global_best_fitness, avg_fitness_history(gen), sum(global_best_chromosome));
    
    % آماده‌سازی ماتریس برای نسل جدید
    new_population = zeros(size(population));
        % تورنومنت
    % جفت‌ها و والدین بر اساس شایستگی‌شان در تورنمنت‌های ۳ تایی انتخاب می‌شوند.
    mating_pool = zeros(size(population));
    for i = 1:pop_size
        mating_pool(i, :) = tournament_selection(population, fitness_values, tournament_size);
    end
        %مرحله ترکیب و تقطیع
    % جفت والدین دو به دو با احتمال pc ترکیب شده و فرزندان جدید را می‌سازند.
    for i = 1:2:pop_size
        if i+1 <= pop_size
            [child1, child2] = crossover(mating_pool(i, :), mating_pool(i+1, :), pc);
            new_population(i, :) = child1;
            new_population(i+1, :) = child2;
        else
            new_population(i, :) = mating_pool(i, :);
        end
    end
    % بیت فلیپ
    % به ازای تک‌تک بیت‌های فرزندان، با احتمال pm (برابر 0.1) بیت معکوس می‌شود.
    for i = 1:pop_size
        new_population(i, :) = mutate(new_population(i, :), pm);
        
        % مجدداً کنترل محدودیت کروموزوم تمام صفر پس از وقوع جهش
        if sum(new_population(i, :)) == 0
            new_population(i, randi(num_features)) = 1;
        end
    end
    % برای اینکه بهترین عضو کشف‌شده در طول تکامل با جهش یا ترکیب‌های بد از بین نرود،
    % آن را مستقیماً به نسل جدید منتقل کرده و در جایگاه اول جمعیت قرار می‌دهیم.
    new_population(1, :) = global_best_chromosome;
    population = new_population; % جایگزینی جمعیت قدیمی با جمعیت نسل جدید
end
disp(' نتایج نهایی الگوریتم ژنتیک');
fprintf('۱. بهترین کروموزوم پیدا شده  [%s]\n', num2str(global_best_chromosome));
fprintf('۲. بهترین مقدار برازندگی: %.4f\n', global_best_fitness);
% پیدا کردن ایندکس ویژگی‌های انتخاب شده (ستون‌هایی که مقدار ژن آن‌ها ۱ است)
selected_indices = find(global_best_chromosome == 1);
fprintf('۳. شماره ویژگی‌های انتخاب شده: %s\n', num2str(selected_indices));
fprintf('   تعداد کل ویژگی‌های انتخاب شده: %d از %d ویژگی\n', length(selected_indices), num_features);
% رسم نمودار همگرایی بهترین و میانگین برازندگی در طول نسل‌ها
figure;
plot(1:num_generations, best_fitness_history, 'r-', 'LineWidth', 2); hold on;
plot(1:num_generations, avg_fitness_history, 'b--', 'LineWidth', 1.5);
title('روند همگرایی الگوریتم ژنتیکی');
xlabel('Generation');
ylabel(' Fitness)');
legend(' (Best)', ' (Average)', 'Location', 'best');
grid on;
% (Fitness Evaluation Function)
% ورودی: یک کروموزوم باینری و مجموعه داده‌های آموزش و تست
% خروجی: مقدار برازندگی محاسبه شده بر اساس فرمول تصویر ۲
function fitness = evaluate_fitness(chromosome, X_train, y_train, X_test, y_test)
    selected_features = find(chromosome == 1); % پیدا کردن ویژگی‌های فعال (ژن‌های ۱)
        % اگر کروموزوم به هر دلیلی خالی باشد، برازندگی آن صفر ارزیابی می‌شود
    if isempty(selected_features)
        fitness = 0;
        return;
    end
        % فیلتر کردن ماتریس ویژگی‌ها بر اساس ژن‌های فعال کروموزوم
    X_train_selected = X_train(:, selected_features);
    X_test_selected  = X_test(:, selected_features);
      knn_model = fitcknn(X_train_selected, y_train, 'NumNeighbors', 3);
        %  پیش‌بینی کلاس داده‌های تست غیردیده‌شده
    y_pred = predict(knn_model, X_test_selected);
    %  محاسبه دقتAccuracy به عنوان نسبت پیش‌بینی‌های درست به کل داده‌های تست
    accuracy = sum(y_pred == y_test) / length(y_test);
    % هرچه تعداد ویژگی‌ها کمتر باشد، جریمه کمتر شده و شایستگی کروموزوم بالاتر می‌رود.
    % Fitness = Accuracy - 0.01 * (تعداد ویژگی‌های انتخاب شده)
    fitness = accuracy - 0.01 * length(selected_features);
end
% Tournament Selection Function
% ورودی: جمعیت کل، مقادیر شایستگی و اندازه تورنمنت
% خروجی: یک والد شایسته انتخاب شده
function selected_parent = tournament_selection(population, fitness_values, tournament_size)
    pop_size = size(population, 1);
    % انتخاب تصادفی تعدادی از اعضا (برابر با tournament_size) برای شرکت در رقابت
    candidates = randperm(pop_size, tournament_size);
    % مقایسه برازندگی شرکت‌کنندگان و استخراج ایندکس فرد برنده
    [~, best_idx] = max(fitness_values(candidates));
    % بازگرداندن کروموزوم فرد برنده تورنمنت به عنوان والد
    selected_parent = population(candidates(best_idx), :);
end
% Single-Point Crossover Function
% ورودی: دو والد و احتمال ترکیب pc
% خروجی: دو فرزند ایجاد شده
function [child1, child2] = crossover(parent1, parent2, pc)
    num_features = length(parent1);
    if rand < pc
        % اگر عدد تصادفی کوچکتر از pc (برابر 0.8) باشد، جفت‌گیری انجام می‌شود
        % انتخاب یک نقطه برش تصادفی بین ژن‌ها
        cp = randi([1, num_features - 1]);
        % تبادل ژن‌ها از نقطه برش به بعد برای تولید دو فرزند
        child1 = [parent1(1:cp), parent2(cp+1:end)];
        child2 = [parent2(1:cp), parent1(cp+1:end)];
    else
        % در غیر این صورت، فرزندان دقیقاً کپی والدین خواهند بود
        child1 = parent1;
        child2 = parent2;
    end
end
% Bit-Flip 
% ورودی: یک کروموزوم و احتمال جهش pm
% خروجی: کروموزوم جهش یافته
function mutated = mutate(chromosome, pm)
    mutated = chromosome;
    for j = 1:length(chromosome)
        % به ازای تک‌تک ژن‌ها، بررسی می‌شود که آیا جهش رخ می‌دهد یا خیر
        if rand < pm
            % اگر عدد تصادفی کمتر از pm (برابر 0.1) باشد، بیت معکوس می‌شود
            mutated(j) = 1 - mutated(j); 
        end
    end
end
