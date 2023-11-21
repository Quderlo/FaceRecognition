import face_recognition


def encode_frame(frame, face_locations, i):
    return face_recognition.face_encodings(frame, [face_locations[i]])[0]


def parse_face(frame):
    # Поиск лиц на фото model="cnn"
    return face_recognition.face_locations(frame)
