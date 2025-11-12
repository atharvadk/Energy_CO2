from flask import Flask, jsonify, request
import mysql.connector
from mysql.connector import Error
import config

app = Flask(__name__)

# -------------------------------
# Database Connection Function
# -------------------------------
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=config.MYSQL_HOST,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DB
        )
        return connection
    except Error as e:
        print("❌ Error while connecting to MySQL:", e)
        return None


# -------------------------------
# 1️⃣ Home Route
# -------------------------------
@app.route('/')
def index():
    return jsonify({"message": "Welcome to Energy–CO₂ Flask API"})


# -------------------------------
# 2️⃣ GET Route – Fetch all records
# -------------------------------
@app.route('/data', methods=['GET'])
def get_all_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM energy_data LIMIT 10;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(rows)


# -------------------------------
# 3️⃣ POST Route – Insert new record
# -------------------------------
@app.route('/data', methods=['POST'])
def add_record():
    data = request.get_json()
    country = data.get('country')
    year = data.get('year')
    energy_consumption = data.get('energy_consumption')
    co2_emission = data.get('co2_emission')

    conn = get_db_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO energy_data (country, year, energy_consumption, co2_emission)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(insert_query, (country, year, energy_consumption, co2_emission))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Record added successfully!"})


# -------------------------------
# 4️⃣ GET Route – Single record by country
# -------------------------------
@app.route('/data/<country>', methods=['GET'])
def get_country_data(country):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM energy_data WHERE country = %s;", (country,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    if result:
        return jsonify(result)
    else:
        return jsonify({"message": f"No data found for {country}"}), 404


# -------------------------------
# 5️⃣ DELETE Route – Delete record
# -------------------------------
@app.route('/data/<int:id>', methods=['DELETE'])
def delete_record(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM energy_data WHERE id = %s;", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"message": f"Record {id} deleted successfully!"})


# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
