# ===================== IMPORT LIBRARY =====================
import pandas as pd
import numpy as np
import math
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression

# ===================== SIDEBAR MENU =====================
menu = st.sidebar.selectbox("📌 Navigasi:", ('🏠 Beranda', '📂 Data Mentah', '🔍 Fitur Teknikal', '🧼 Pembersihan Data', '📈 Prediksi'))

# ===================== INPUT USER =====================
emiten = st.text_input('📝 Masukkan kode emiten (contoh: BBCA.JK):')
startTanggal = '2015-01-01'
endTanggal = st.text_input('📅 Masukkan tanggal akhir (format: YYYY-MM-DD):')

if emiten == "" or endTanggal == "":
    st.info("Silakan masukkan kode emiten dan tanggal terlebih dahulu untuk memulai analisis.")
else:
    # ===================== UNDUH DATA =====================
    data = yf.download(emiten, start=startTanggal, end=endTanggal)

    # ===================== FITUR TEKNIKAL =====================
    data['SPV'] = ((data['High'] - data['Low']) / data['Close']) * 100  # Spread Volatility
    data['CHG'] = ((data['Close'] - data['Open']) / data['Open']) * 100  # Daily Change
    data1 = data[['SPV', 'Close', 'CHG', 'Volume']]
    data1.fillna(value=-99999, inplace=True)

    # ===================== MENU: BERANDA =====================
    if menu == '🏠 Beranda':
        st.title("📊 Aplikasi Prediksi Harga Saham")
        st.markdown(f"""
        Aplikasi ini dirancang untuk memprediksi harga saham berdasarkan indikator teknikal menggunakan pendekatan **Machine Learning** seperti Support Vector Regression (SVR) dan Linear Regression.

        **Kode Emiten:** `{emiten.upper()}`  
        **Periode Data:** `{startTanggal}` hingga `{endTanggal}`

        ---
        - Putwal Arjuna Murti (220210502029)
        """)

    # ===================== MENU: DATA MENTAH =====================
    elif menu == '📂 Data Mentah':
        st.subheader("📄 Data Historis Saham")
        st.write(data)
        st.line_chart(data['Close'])

    # ===================== MENU: FITUR TEKNIKAL =====================
    elif menu == '🔍 Fitur Teknikal':
        st.subheader("📈 Feature Engineering")
        st.markdown("""
        - **SPV (Spread Volatility)**: Indikator volatilitas harian berdasarkan selisih antara harga tertinggi dan terendah relatif terhadap harga penutupan.
        - **CHG (Change)**: Persentase perubahan harga harian dari harga pembukaan ke penutupan.
        """)
        st.dataframe(data1)

    # ===================== MENU: CLEANING =====================
    elif menu == '🧼 Pembersihan Data':
        st.subheader("🧹 Data Cleaning")
        st.markdown("""
        Missing values telah digantikan dengan nilai ekstrem `-99999` agar dapat diabaikan oleh model saat pelatihan.
        """)
        st.write(data1)

    # ===================== MENU: PREDIKSI =====================
    elif menu == '📈 Prediksi':
        st.subheader("📉 Prediksi Harga Saham")

        # Output future prediction days
        jml_OutputPrediksi = int(math.ceil(0.01 * len(data1)))
        data1['OutputPrediksi'] = data1['Close'].shift(-jml_OutputPrediksi)
        st.success(f"Model akan memprediksi {jml_OutputPrediksi} hari ke depan berdasarkan pola historis.")

        # Persiapan dataset
        X = np.array(data1.drop(columns=['OutputPrediksi']))
        X = preprocessing.scale(X)
        X_Prediksi = X[-jml_OutputPrediksi:]
        X = X[:-jml_OutputPrediksi]
        data1.dropna(inplace=True)
        y = np.array(data1['OutputPrediksi'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # ===================== MODEL: SVR =====================
        st.markdown("🔵 **Support Vector Regression (SVR)**")
        model_svr = svm.SVR()
        model_svr.fit(X_train, y_train)
        akurasi_svr = model_svr.score(X_test, y_test)
        st.write(f"Akurasi Model SVR: **{akurasi_svr:.4f}**")
        prediksi_svr = model_svr.predict(X_Prediksi)
        st.write("Hasil prediksi SVR:")
        st.write(prediksi_svr)

        # ===================== MODEL: LINEAR REGRESSION =====================
        st.markdown("🟢 **Linear Regression**")
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        akurasi_lr = model_lr.score(X_test, y_test)
        st.write(f"Akurasi Model Linear Regression: **{akurasi_lr:.4f}**")
        prediksi_lr = model_lr.predict(X_Prediksi)
        st.write("Hasil prediksi Linear Regression:")
        st.write(prediksi_lr)
        st.line_chart(prediksi_lr)

        # ===================== VISUALISASI =====================
        st.markdown("📊 **Visualisasi Prediksi vs Data Aktual**")
        data1['prediksi'] = np.nan
        lastDate = data1.iloc[-1].name
        lastSecond = lastDate.timestamp()
        oneDay = 86400
        nextSecond = lastSecond + oneDay

        for pred in prediksi_lr:
            nextDate = datetime.datetime.fromtimestamp(nextSecond)
            nextSecond += 86400
            data1.loc[nextDate] = [np.nan for _ in range(len(data1.columns)-1)] + [pred]

        fig, ax = plt.subplots()
        data1['Close'].plot(ax=ax, label='Harga Aktual')
        data1['prediksi'].plot(ax=ax, label='Prediksi', linestyle='--')
        ax.set_title(f"Perbandingan Harga Aktual vs Prediksi ({emiten.upper()})")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga Saham")
        ax.legend()
        st.pyplot(fig)

        st.warning("🔍 Evaluasi model berdasarkan akurasi & tren hasil prediksi sebelum mengambil keputusan.")
