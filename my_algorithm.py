import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, f1_score

def algorithm(model, X_train, y_train, X_test, y_test, Accuracy):
  model.fit(X_train, y_train) # huấn luyện mô hình
  prediction = model.predict(X_test) # dự đoán nhãn của tập X_test

  # In ra ma trận nhầm lẫn
  print('confusion matrix')
  # kích thước cm là 2x2, với 2 hàng và 2 cột với các giá trị là số nguyên
  cm = confusion_matrix(y_test,prediction)

  # Tạo nhãn cho ma trận nhầm lẫn
  group_names = ["True Negative","False Positive",'False Negative',"True Positive"] # Khởi tạo tên cho các nhóm
  # Tính toán số lượng mẫu
  # Hàm flatten() được sử dụng để chuyển ma trận thành mảng 1 chiều
  # '{0:0.0f}'.format(value): Dùng để định dạng các giá trị này thành chuỗi dạng số nguyên, không có phần thập phân
  group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()] 
  # Tính toán tỷ lệ phần trăm
  # {0:.2%}.format(value): Dùng để định dạng các giá trị này thành chuỗi dạng số thập phân, với 2 chữ số sau dấu phẩy
  # np.sum(cm): Tính tổng các phần tử trong ma trận
  # cm.flatten()/np.sum(cm): Tính tỷ lệ phần trăm của từng phần tử trong ma trận
  group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
  # f"{v1}\n{v2}\n{v3}" : Dùng để kết hợp các giá trị v1, v2, v3 thành một chuỗi
  # f là viết tắt của format
  # f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages): Duyệt qua từng phần tử trong group_names, group_counts, group_percentages
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  # Chuyển labels thành mảng numpy kích thước 2x2
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')

  plt.title("Confusion Matrix")
  plt.xlabel('predicted label')
  plt.ylabel('True label')
  plt.show()

  print(cm)
  # In ra báo cáo phân loại
  # classification_report(y_test,prediction): Tạo báo cáo phân loại dựa trên nhãn thực tế và nhãn dự đoán như thế nào? 
  # Báo cáo phân loại bao gồm precision, recall, f1-score và support, tính các thuộc tính này như thế nào? 
  # precision = TP / (TP + FP); recall = TP / (TP + FN); f1-score = 2 * (precision * recall) / (precision + recall)
  # support: số lượng mẫu thực tế của lớp đó
  # accuracy = (TP + TN) / (TP + TN + FP + FN)
  # macro avg = (precision_0 + precision_1) / 2
  # weighted avg = (precision_0 * support_0 + precision_1 * support_1) / (support_0 + support_1)
  print(classification_report(y_test,prediction))

  weighted_f1 = f1_score(y_test, prediction, average='weighted')

  final_score = weighted_f1*100
  print('weighted_f1_score : ' , final_score)
  Accuracy.append(weighted_f1*100)

# ví dụ
'''
[[106   1]
 [  5  59]]
              precision    recall  f1-score   support

         0.0       0.95      0.99      0.97       107
         1.0       0.98      0.92      0.95        64

    accuracy                           0.96       171
   macro avg       0.97      0.96      0.96       171
weighted avg       0.97      0.96      0.96       171


Có TN = 106, TP = 59, FP = 1, FN = 5
Có precision(0) = TN / TN + FN = 106/(106 + 5) = 0.95; recall(0) = TN / TN + FP = 106/(106 + 1) = 0.99; f1-score(0) = 2 * (precision(0) * recall(0)) / (precision(0) + recall(0)) = 0.97
Có precision(1) = TP / TP + FP = 59/(59 + 1) = 0.98; recall(1) = TP / TP + FN = 59/(59 + 5) = 0.92; f1-score(1) = 2 * (precision(1) * recall(1)) / (precision(1) + recall(1)) = 0.95

weighted avg = (precision(0) * support(0) + precision(1) * support(1)) / (support(0) + support(1)) = (0.95 * 107 + 0.98 * 64) / (107 + 64) = 0.96
'''