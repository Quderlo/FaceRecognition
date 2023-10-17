import cv2
from parse_and_encode import parse_face
from parse_and_encode import encode_frame
from check_encode_in_database import compare_encodings
import PATH
import Constants
import os
import numpy as np

# Загрузка изображения
if __name__ == '__main__':
    # Загрузка видео
    video = cv2.VideoCapture(PATH.video)
    frame_count = 0

    while True:
        # Чтение кадра из видео
        ret, frame = video.read()
        frame_count += 1

        # Прекращение работы, если кадр не удалось прочитать
        if not ret:
            break

        # Пропускаем кадры, если это необходимо
        if frame_count % Constants.frame_skip != 0:
            continue

        face_locations = parse_face(frame)  # Поиск лиц в кадре

        # Если обнаружены лица, обрабатываем их
        if len(face_locations) > 0:
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Получение кодировки лица
                face_encoding = encode_frame(frame, face_locations, i)

                if Constants.debug:
                    # Обводка лиц
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Сохранение лица
                    face_img = frame[top:bottom, left:right]
                    cv2.imwrite(f"parse/face_{frame_count}_face_{i}.jpg", face_img)

                    if not os.path.exists("encodings"):
                        os.makedirs("encodings")
                    np.save(f"encodings/face_{frame_count}_face_{i}.npy", face_encoding)

                # Сравнение кодировки с уже имеющимися в базе
                compare_encodings(face_encoding, PATH.database, Constants.threshold)

        if Constants.show_video:
            # Отображение видео с обведенными лицами
            cv2.imshow("Result", frame)

    # Закрытие видео и окна вывода
    video.release()
    cv2.destroyAllWindows()
