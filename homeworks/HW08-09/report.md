# HW08-09 – PyTorch MLP: регуляризация и оптимизация обучения

## 1. Кратко: что сделано
- Выбран датасет: **KMNIST**.
- В части A сравнивались: базовый MLP, +Dropout, +BatchNorm, +EarlyStopping.
- В части B исследованы: слишком большой LR, слишком маленький LR, SGD+momentum + weight decay.

## 2. Среда и воспроизводимость
- Python: 3.14
- torch / torchvision: см. вывод в ноутбуке
- Устройство: CPU/GPU
- Seed: 42
- Запуск: открыть `HW08-09.ipynb` → Run All

## 3. Данные
- Датасет: KMNIST
- Разделение: train/val = 80/20, test — стандартный
- Трансформации: ToTensor()
- KMNIST содержит 10 классов, изображения 28×28, задача средней сложности.

## 4. Базовая модель и обучение
- MLP: 2 скрытых слоя по 256 нейронов, ReLU
- Loss: CrossEntropyLoss
- Optimizer (часть A): Adam (lr=1e-3)
- Batch size: 128
- Epochs: до 20 (E1–E3), до 50 (E4)
- EarlyStopping: patience=5

## 5. Часть A (S08): регуляризация
- **E1**: базовый MLP, без Dropout/BatchNorm  
- **E2**: как E1 + Dropout(p=0.3)  
- **E3**: как E1 + BatchNorm  
- **E4**: лучший из (E2/E3) + EarlyStopping  

## 6. Часть B (S09): LR, оптимизаторы, weight decay
- **O1**: Adam, lr=1e-1 (слишком большой)
- **O2**: Adam, lr=1e-5 (слишком маленький)
- **O3**: SGD+momentum=0.9, weight_decay=1e-4, lr=1e-2

## 7. Результаты
- Таблица: `./artifacts/runs.csv`
- Лучшая модель: `./artifacts/best_model.pt`
- Конфиг: `./artifacts/best_config.json`
- Кривые лучшего прогона: `./artifacts/figures/curves_best.png`
- Кривые LR: `./artifacts/figures/curves_lr_extremes.png`

Краткая сводка:
- Лучший эксперимент части A: **E4**
- Лучшая val_accuracy: **(вставить значение)**
- Итоговая test_accuracy: **(вставить значение)**
- O1: loss нестабилен, accuracy падает
- O2: обучение почти не движется
- O3: SGD+momentum сходится медленнее Adam, но стабильнее

## 8. Анализ
- Dropout и BatchNorm уменьшают переобучение, что видно по меньшему разрыву train/val.
- EarlyStopping остановил обучение раньше, сохранив лучший чекпойнт.
- O1 показывает расходимость из-за слишком большого шага.
- O2 почти не обучается из-за слишком маленького шага.
- SGD+momentum + weight decay даёт более гладкие кривые.
- Лучший конфиг (E4) оптимален для KMNIST.

## 9. Итоговый вывод
- Базовый конфиг: MLP(256,256) + BatchNorm или Dropout + EarlyStopping.
- Для улучшения можно попробовать:  
  1) увеличить модель  
  2) добавить scheduler (StepLR/ReduceLROnPlateau)

