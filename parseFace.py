import cv2
import face_recognition
import Constants
import PATH
from check_add_encode import compare_encodings


def process_frame(frame, face_locations, frame_count, debug=False):
    # Проверка, нужно ли сохранять кодировки и изображения
    save_data = debug

    # Очерчивание контуров лиц на кадре
    for i, (top, right, bottom, left) in enumerate(face_locations):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if save_data:
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(f"parse/face_{frame_count}_face_{i}.jpg", face_img)

            face_encoding = face_recognition.face_encodings(frame, [face_locations[i]])[0]
            compare_encodings(face_encoding, PATH.database, Constants.threshold, debug)


def parse_face(path_to_video, debug=False):
    # Загрузка видео
    video = cv2.VideoCapture(path_to_video)

    frame_count = 0

    while True:
        # Чтение кадра из видео
        ret, frame = video.read()

        # Прекращение работы, если кадр не удалось прочитать
        if not ret:
            break

        frame_count += 1

        # Пропускаем кадры, если это необходимо
        if frame_count % Constants.frame_skip != 0:
            continue

        # Обнаружение лиц на кадре
        face_locations = face_recognition.face_locations(frame)

        # Если обнаружены лица, обрабатываем их
        if len(face_locations) > 0:
            process_frame(frame, face_locations, frame_count, debug)

        if debug:
            # Отображение видео с обведенными лицами
            cv2.imshow("Result", frame)

        # Прекращение работы при нажатии клавиши q
        if cv2.waitKey(1) == ord("q"):
            break

    # Закрытие видео и окна вывода
    video.release()
    cv2.destroyAllWindows()