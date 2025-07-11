Đếm Người Ra/Vào Xe Buýt và Cảnh Báo

Trong bối cảnh nhu cầu sử dụng phương tiện giao thông công cộng ngày càng tăng, việc giám sát và kiểm soát số lượng hành khách trên xe buýt trở nên vô cùng quan trọng. Đề tài này hướng đến việc xây dựng một hệ thống thông minh sử dụng thị giác máy tính nhằm phát hiện, đếm số người ra/vào xe buýt và cảnh báo khi quá tải. Qua đó, góp phần đảm bảo an toàn, tối ưu vận hành và nâng cao trải nghiệm của hành khách.

📌 Mục tiêu đề tài

- Xây dựng hệ thống ứng dụng thị giác máy tính vào việc:

- Nhận diện và theo dõi người trong video.

- Đếm số người lên và xuống xe buýt tại các điểm dừng.

- Cảnh báo khi số lượng hành khách vượt quá giới hạn cho phép.

🛠 CÔNG NGHỆ SỬ DỤNG

- YOLOv5: Nhận diện người theo thời gian thực.

- OpenCV: Xử lý video, khung hình, vẽ bounding box.

- Python: Lập trình xử lý logic, đếm người, cảnh báo.

- NumPy, Matplotlib: Phân tích dữ liệu và trực quan hóa (tuỳ chọn).

- Âm thanh & màu sắc cảnh báo khi vượt quá ngưỡng giới hạn.

🔁 Chu trình hoạt động cơ bản

- Đọc video đầu vào (camera hoặc file).

- Nhận diện người bằng YOLOv8 hoặc HOG.

- Xác định vị trí và hướng di chuyển của từng người.

- Kiểm tra người có đi qua vùng cửa (vào hay ra).

- Cập nhật bộ đếm người vào/ra và tổng số hiện tại.

- Nếu tổng số người > ngưỡng → phát cảnh báo trực quan.

💡 Mục tiêu mở rộng trong tương lai

- Tích hợp hệ thống với camera thời gian thực trên xe buýt thật.

- Áp dụng thêm mô hình theo dõi người (DeepSORT, ByteTrack…).

- Kết hợp với cảnh báo đám đông, chen lấn, hành vi nguy hiểm.

- Xuất báo cáo thống kê hành khách tự động theo tuyến/ngày.

- Ứng dụng trên nhiều loại phương tiện công cộng khác nhau.

👨‍💻 Nhóm thực hiện

- Nguyễn Hoài Nam -1571020183

- Nguyễn Tuấn Anh - 1571020013

- Hoàng Văn Lâm - 1571020148
