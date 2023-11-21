import cv2
from ParseAndEncode import parse_face, encode_frame
from CheckEncodeInDatabase import compare_encodings
import PATH
import Constants
import os
import numpy as np
import multiprocessing
import time


def process_frame(frame_count):
    video = cv2.VideoCapture(PATH.video)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Устанавливаем начальный кадр

    for _ in range(Constants.frame_skip):
        # Чтение кадра из видео
        ret, frame = video.read()
        frame_count += 1

        # Прекращение работы, если кадр не удалось прочитать
        if not ret:
            break

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

        # Прекращение работы при нажатии клавиши q
        if cv2.waitKey(1) == ord("q"):
            break

    # Закрытие видео
    video.release()


if __name__ == "__main__":
    start_time = time.time()
    video = cv2.VideoCapture(PATH.video)

    # Получение общего количества кадров видео
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Задаем количество процессов
    num_processes = Constants.processors_count

    frames_per_process = total_frames // num_processes

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_frame, range(0, total_frames, Constants.frame_skip))
    close.co
    end_time = time.time()

    elapsed_time = end_time - start_time
    print('Elapsed time: ', elapsed_time)
    # Закрытие окна вывода
    cv2.destroyAllWindows()
