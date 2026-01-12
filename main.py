import sys
import os
import logging
# Tắt log thừa của Paddle
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR
import numpy as np

class MedicalOCR:
    def __init__(self):
        print(">> Đang khởi tạo model...", end="\r")
        # use_angle_cls=True: Tự động xoay ảnh
        # lang='vi': Hỗ trợ tiếng Việt
        self.ocr = PaddleOCR(use_angle_cls=True, lang='vi')
        print(">> Khởi tạo model xong!       ")

    def process_image(self, image_path):
        if not os.path.exists(image_path):
            return f"LỖI: Không tìm thấy file ảnh tại: {image_path}"

        print(f">> Đang đọc ảnh: {image_path}")
        
        # Chạy OCR
        try:
            # Phiên bản mới chỉ cần truyền đường dẫn
            result = self.ocr.ocr(image_path)
        except Exception as e:
            return f"Lỗi nội bộ PaddleOCR: {str(e)}"

        # DEBUG: In ra cấu trúc dữ liệu để kiểm tra
        if not result or result[0] is None:
            return "Không tìm thấy văn bản nào trong ảnh."
        
        # Lấy danh sách kết quả (thường nằm ở phần tử đầu tiên)
        raw_data = result[0]
        
        # In thử 1 dòng dữ liệu đầu tiên để xem cấu trúc (Debug)
        print("-" * 30)
        print("Cấu trúc dữ liệu mẫu (Item 0):")
        print(raw_data[0]) 
        print("-" * 30)

        # Bắt đầu xử lý ghép dòng
        try:
            sorted_lines = self.smart_sort_lines(raw_data)
            return "\n".join(sorted_lines)
        except Exception as e:
            # Nếu lỗi ở đây, ta sẽ biết chính xác tại sao
            return f"Lỗi khi xử lý ghép dòng: {str(e)}"

    def smart_sort_lines(self, boxes, y_threshold=10):
        # Chuyển đổi dữ liệu sang dạng chuẩn để tránh lỗi Index
        clean_boxes = []
        for item in boxes:
            # item thường là [coordinates, (text, confidence)]
            coords = np.array(item[0]) # Chuyển tọa độ thành numpy array
            text_info = item[1]
            clean_boxes.append({'coords': coords, 'text': text_info[0]})

        # 1. Sort từ trên xuống dưới theo Y (góc trên-trái)
        # coords[0][1] là Y của điểm đầu tiên
        clean_boxes.sort(key=lambda x: x['coords'][0][1])

        lines = []
        if not clean_boxes:
            return lines

        current_line = [clean_boxes[0]]
        last_y = clean_boxes[0]['coords'][0][1]

        for i in range(1, len(clean_boxes)):
            box = clean_boxes[i]
            curr_y = box['coords'][0][1]

            if abs(curr_y - last_y) <= y_threshold:
                current_line.append(box)
            else:
                # Hết dòng -> Sort cột theo X (từ trái qua phải)
                current_line.sort(key=lambda x: x['coords'][0][0])
                # Ghép text
                line_text = " | ".join([b['text'] for b in current_line])
                lines.append(line_text)
                
                # Reset
                current_line = [box]
                last_y = curr_y

        # Xử lý dòng cuối
        if current_line:
            current_line.sort(key=lambda x: x['coords'][0][0])
            line_text = " | ".join([b['text'] for b in current_line])
            lines.append(line_text)

        return lines

if __name__ == "__main__":
    # Mặc định lấy tên ảnh
    img_path = 'don_thuoc.jpg'
    if len(sys.argv) > 1:
        img_path = sys.argv[1]

    engine = MedicalOCR()
    print(engine.process_image(img_path))