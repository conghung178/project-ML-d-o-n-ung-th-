# data_split.py

from sklearn.model_selection import train_test_split

def split_data(X, y):
    """
    Hàm chia dữ liệu thành tập huấn luyện và tập kiểm tra.
    
    Parameters:
    - X: Dữ liệu đầu vào (features)
    - y: Dữ liệu đầu ra (labels)
    - test_size: Tỉ lệ chia tập kiểm tra (default là 0.3)
    - random_state: Giá trị để đảm bảo chia dữ liệu tái lặp được (default là 30)
    - stratify: Đảm bảo chia dữ liệu đồng đều theo phân phối nhãn (default là None)

    Returns:
    - X_train: Tập huấn luyện
    - X_test: Tập kiểm tra
    - y_train: Nhãn tập huấn luyện
    - y_test: Nhãn tập kiểm tra
    """
    return train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

