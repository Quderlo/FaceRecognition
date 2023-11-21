import face_recognition
import Constants
import pickle
from DatabaseConnection import connection as conn


def compare_encodings(face_encoding, database_path, threshold):
    with conn.cursor() as cursor:
        cursor.execute('SELECT face_encoding FROM face_encodings_table;')
        rows = cursor.fetchall()
        database_encodings = [pickle.loads(row[0]) for row in rows]

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
        with conn.cursor() as cursor:
            cursor.execute('INSERT INTO face_encodings_table (face_encoding) VALUES (%s);',
                           (pickle.dumps(face_encoding),))
            conn.commit()
            if Constants.debug:
                print(f"Similarity: {similarity:.2f}% - Face saved to the database")
    else:
        if Constants.debug:
            print(f"Similarity: {similarity:.2f}% - Face not saved to the database")
