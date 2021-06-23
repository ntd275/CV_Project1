# BTL_CV

## Set up
#### Cài đặt môi trường
- Yêu cầu: cài đặt python3(hiện tại em sử dụng python3.6)
- Cài đặt các package và thư viện liên quan
```bash
pip install -r requirements.txt
```
## Chạy chương trình
### Chạy với các ảnh có sẵn trong thư mục images
B1: Mở file threshold.py tại dòng:
```
defaultImage = listImage[2]
config = config2
```
B2: Sửa defaultImage bằng image muốn chạy trong list 0 -> 7   
VD:
```
defaultImage = listImage[0]
```
Sửa config bằng config có sẵn gồm: config013, config2, config4, config56, config7 tùy vào ảnh chọn ở trên.      
VD: Bên trên sử dụng ảnh 0 trong list nên sửa:
```
config=config013
```
B3: Chạy lệnh
```
python threshold.py
```
### Chạy với ảnh tùy chọn và config tùy chọn
B1: Mở file threshold.py tại dòng:
```
config = config2
```
Sửa config bằng config tùy chọn     
VD:
```
config={
    "BlurKsize": 1, #2*value+1
    "BlockSize": 200, #2*value+1
    "C": 0,
    "OpeningKsize": 4,
    "Invert": False,
    "Sinus" : False,
    "SinusThreshold": 230,
}
```
Giá trị BlurKsize là kennel size khi sử dụng median filter do nó là số lẻ nên giá trị sử dụng sẽ là giá trị trong config nhân 2 cộng 1. VD trên set "BlurKsize" : 1 thì giá trị khi chạy chương trình là 1*2+1 = 3.

Giá trị BlockSize là block size khi sử dụng adaptiveThreshold tương tự BlurKsize nó là số lẻ nên giá trị sử dụng sẽ là giá trị trong config nhân 2 cộng 1. VD trên set "BlockSize" : 200 thì giá trị khi chạy chương trình là 200*2+1 = 401.

Giá trị C là giá trị để lấy ngưỡng trong hàm adaptiveThreshold: threshold = giá trị trung bình của khu vực trừ đi giá trị C.   


Giá trị OpeningKsize là kennel size khi sử dụng morphologyEx. Khi được đặt là a thì kennel size là a x a.

Giá trị Invert là ảnh có nền sáng hơn vật không. Nếu đặt là True thì bước lấy adaptiveThreshold sẽ đảo ngược ảnh nhị phân lại để nền đen và vật màu trắng.

Giá trị Sinnus là ảnh có nhiễu hình sin không nếu đặt là True ảnh sẽ được lọc các tần số có cường độ cao hơn giá trị SinusThreshold trong config( sau khi được chuẩn hóa trên thang 0 - 255)


B3: Chạy lệnh
```
python threshold.py --image=[đường dẫn đến ảnh cần xử lý]
```

Sau khi chạy có thể hiệu chỉnh config bằng cách kéo các trackbar để được kết quả tốt nhất.