# import numpy as np
# from flask import Flask, request, render_template,redirect, url_for,session
# import pickle
# import os
# from math import log
# import pandas as pd
# import seaborn as sns 
# import matplotlib.pyplot as plt


# # Flask uygulamasını oluştur
# app = Flask(__name__)


# # Modeli yükle
# model_path = os.path.join("templates", "model.pkl")
# model = pickle.load(open(model_path, "rb"))

# @app.route("/")
# def home():
#     return render_template("index.html")



# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         # Form verilerini al
#         float_features = [float(x) for x in request.form.values()]
#         # Değerlerin logaritmasını al
#         log_features = [log(x) for x in float_features]
#         features = [np.array(log_features)]
#         # Tahmini hesapla
#         prediction = model.predict(features)
#         # Sonucu template'e gönder
#         return render_template("predict.html", prediction_text="The predicted value is {:.2f}".format(prediction[0]))
#     return render_template("predict.html")

# # @app.route("/upload", methods=["GET", "POST"])
# # def upload():
# #     if request.method == "POST":
# #         file = request.files["file"]
# #         if file.filename.endswith(".csv"):
# #             df = pd.read_csv(file)
# #             return render_template("upload.html", tables=[df.to_html(classes="data")], titles=df.columns.values)
# #     return render_template("upload.html")

# @app.route("/upload", methods=["GET", "POST"])
# def upload():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file and file.filename.endswith(".csv"):
#             df = pd.read_csv(file)
#             message = "File uploaded successfully."
#             return render_template("upload.html", message=message)
#         else:
#             message = "Please upload a valid CSV file."
#             return render_template("upload.html", message=message)
#     return render_template("upload.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file and file.filename.endswith(".csv"):
#             # Read CSV file
#             df = pd.read_csv(file)
#             # Perform data analysis
#             summary_statistics = df.describe().to_dict()
#             correlation_matrix = df.corr().to_dict()
#             # Create plot
#             sns.pairplot(df)
#             plt.savefig(os.path.join("static", "pairplot.png"))
#             plt.close()
#             # Render results
#             return render_template("analyze.html", 
#                                    summary_statistics=summary_statistics, 
#                                    correlation_matrix=correlation_matrix)
#     return redirect(url_for("home"))

# if __name__ == "__main__":
#     if not os.path.exists("static"):
#         os.makedirs("static")
#     app.run(debug=True)
import numpy as np
from quart import Quart, request, render_template, redirect, url_for
import pickle
import os
from math import log
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Quart uygulamasını oluştur
app = Quart(__name__)

# Modeli yükle
model_path = os.path.join("templates", "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
async def home():
    return await render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
async def predict():
    if request.method == "POST":
        # Form verilerini al
        form_values = await request.form
        float_features = [float(x) for x in form_values.values()]
        # Değerlerin logaritmasını al
        log_features = [log(x) for x in float_features]
        features = [np.array(log_features)]
        # Tahmini hesapla
        prediction = model.predict(features)
        # Sonucu template'e gönder
        return await render_template("predict.html", prediction_text="The predicted value is {:.2f}".format(prediction[0]))
    return await render_template("predict.html")



@app.route("/upload", methods=["GET", "POST"])
async def upload():
    if request.method == "POST":
        files = await request.files
        file = files.get("file")
        if file and file.filename.endswith(".csv"):
            # Dosyayı sunucuya kaydet
            file_path = os.path.join("uploads", file.filename)
            await file.save(file_path)
            # Dosya yüklendikten sonra kullanıcıyı analyze sayfasına yönlendir
            return redirect(url_for("analyze", file=file.filename))
        else:
            message = "Please upload a valid CSV file."
            return await render_template("upload.html", message=message)
    return await render_template("upload.html")

def detect_outliers(df):
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers

@app.route("/analyze", methods=["GET", "POST"])
async def analyze():
    if request.method == "POST":
        # POST isteklerini yönlendirme
        return redirect(url_for("home"))
    
    file_name = request.args.get("file")
    if file_name:
        file_path = os.path.join("uploads", file_name)
        if os.path.exists(file_path):
            # Veri setini yükle
            df = pd.read_csv(file_path)
            # Kategorik ve sayısal sütunları ayır
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(exclude=[np.number])
            
            categorical_summary = {}
            summary_statistics = {}
            correlation_matrix = {}
            outliers = {}

            if not categorical_df.empty:
                # Kategorik sütunlar için frekans tabloları
                categorical_summary = {col: categorical_df[col].value_counts().to_dict() for col in categorical_df.columns}
                # Kategorik sütunların grafiklerini oluştur
                for col in categorical_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(y=col, data=categorical_df, order=categorical_df[col].value_counts().index)
                    plt.title(f"Distribution of {col}")
                    plt.savefig(os.path.join("static", f"categorical_{col}.png"))
                    plt.close()

            if not numeric_df.empty:
                # Sayısal sütunlar için özet istatistikler ve korelasyon matrisi
                summary_statistics = numeric_df.describe().to_dict()
                correlation_matrix = numeric_df.corr().to_dict()
                
                # Aykırı değerleri tespit et
                outliers = detect_outliers(numeric_df)
                
                # Pairplot oluştur
                sns.pairplot(numeric_df)
                plt.savefig(os.path.join("static", "pairplot.png"))
                plt.close()

                # Aykırı değer grafikleri oluştur
                for column, outlier_values in outliers.items():
                    if not outlier_values.empty:
                        plt.figure()
                        sns.boxplot(x=numeric_df[column])
                        for outlier in outlier_values:
                            plt.plot(outlier, 0, 'ro')
                        plt.title(f"Aykiri Değerler - {column}")
                        plt.savefig(os.path.join("static", f"outliers_{column}.png"))
                        plt.close()

            # Sonuçları kullanıcıya göster
            return await render_template("analyze.html", 
                                         summary_statistics=summary_statistics, 
                                         correlation_matrix=correlation_matrix,
                                         categorical_summary=categorical_summary,
                                         outliers=outliers)
    return redirect(url_for("home"))















@app.route("/login", methods=["GET", "POST"])
async def login():
    if request.method == "POST":
        form_values = await request.form
        email = form_values.get('email')
        password = form_values.get('password')
        # Burada giriş doğrulama işlemi yapın
        return redirect(url_for("home"))
    return await render_template("login.html")

@app.route("/forgot-password")
async def forgot_password():
    return await render_template("forgot-password.html")

@app.route("/register")
async def register():
    return await render_template("register.html")






@app.route("/financial_analysis", methods=["GET", "POST"])
async def financial_analysis():
    if request.method == "POST":
        form_values = await request.form
        income = float(form_values.get('income'))
        expense = float(form_values.get('expense'))
        net_profit = income - expense
        return await render_template("finansalanaliz.html", result=f"Net profit is {net_profit:.2f}")
    return await render_template("finansalanaliz.html")









if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)





