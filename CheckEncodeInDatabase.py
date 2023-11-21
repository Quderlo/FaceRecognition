import numpy as np
import face_recognition
import os
from DatabaseConnection import connection as conn
import Constants
import pickle


def compare_encodings(face_encoding, database_path, threshold):

    # Загрузка текущих кодировок лиц из базы данных (другой папки)
    database_encodings = []
    for filename in os.listdir(database_path):
        if filename.endswith(".npy"):
            database_file = os.path.join(database_path, filename)
            database_encoding = np.load(database_file)
            database_encodings.append(database_encoding)

    # Сравнение переданной кодировки лица с кодировками из базы данных
    match = False
    similarity = 0
    for database_encoding in database_encodings:
        # Сравнение кодировок с помощью библиотеки face_recognition
        face_distance = face_recognition.face_distance([database_encoding], face_encoding)[0]
        similarity = (1 - face_distance) * 100
        if similarity > threshold:
            match = True
            break

    # Если кодировка лица не найдена в базе данных, добавляем ее туда
    if not match:
        new_encoding_file = os.path.join(database_path, f"new_face_{len(database_encodings)}.npy")
        np.save(new_encoding_file, face_encoding)
        database_encodings.append(face_encoding)
        if Constants.debug:
            print(f"Similarity: {similarity:.2f}% New face added to the database: {new_encoding_file}")
            print(f"Similarity: {similarity:.2f}% - Face saved to the database")
    else:
        if Constants.debug:
            print(f"Similarity: {similarity:.2f}% - Face not saved to the database")

    # print(type(face_encoding))
    # face_bytes = pickle.dumps(face_encoding)
    #
    # with conn.cursor() as cursor:
    #     cursor.execute('INSERT INTO faces (face_bytes) VALUES (%s);', (face_bytes,))
    #     conn.commit()
    #
    # with conn.cursor() as cursor:
    #     cursor.execute('SELECT face_bytes FROM faces WHERE id = %s;', (1,))
    #     result = cursor.fetchone()
    #     stored_face_bytes = result[0]
    #
    # stored_face_encoding = pickle.loads(stored_face_bytes)
    #
    # if np.array_equal(face_encoding, stored_face_encoding):
    #     print("Значения идентичны")
    # else:
    #     print("Значения не совпадают")