import psycopg2

connection = psycopg2.connect(
    host='localhost',
    port='5432',
    database='db_facerec',
    user='postgres',
    password='123',
)
