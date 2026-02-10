import streamlit as st
import joblib
import os
import requests

st.set_page_config(page_title="Haber Doğrulama Paneli", layout="wide")
# Google Drive'dan büyük dosyayı indirme fonksiyonu
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)

# --- DRIVE AYARLARI ---
# Buraya Google Drive'daki model dosyanın ID'sini yazmalısın
FILE_ID = '1beTkZgnjG_PU5kXtYvt2OEuzgSUoL5w-'
DESTINATION = 'random_forest_model.pkl'

@st.cache_resource
def model_yukle():
    # Model dosyası yoksa Drive'dan indir
    if not os.path.exists(DESTINATION):
        download_file_from_google_drive(FILE_ID, DESTINATION)
    
    # Dosyaları yükle
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load(DESTINATION)
    return vectorizer, model

try:
    cv, rf_model = model_yukle()
except Exception as e:
    st.error(f"HATA: Model yüklenemedi. Detay: {e}")
except FileNotFoundError:
    st.error("HATA: .pkl dosyaları bulunamadı! Lütfen 'vectorizer.pkl' ve 'random_forest_model.pkl' dosyalarının bu klasörde olduğundan emin olun.")
    st.stop()

# 2. Sayfa Tasarımı (HTML bilmeden web sitesi yapıyoruz!)
st.title("🕵️‍♂️ Yalan Haber Dedektörü")
st.write("Aşağıya İngilizce bir haber metni yapıştırın, yapay zeka analiz etsin.")

# Kullanıcıdan metin alma kutusu
user_input = st.text_area("Haber Metnini Buraya Giriniz:", height=150)

# Butona basılınca ne olsun?
if st.button("Analiz Et"):
    if user_input:
        # A. Temizlik (Reuters vb. silme - Modelimiz böyle eğitildi)
        cleaned_text = user_input.replace("Reuters", "").replace("reuters", "")
        
        # B. Sayıya Çevirme
        vectorized_text = cv.transform([cleaned_text])
        
        # C. Tahmin Etme
        prediction = rf_model.predict(vectorized_text)
        probability = rf_model.predict_proba(vectorized_text)
        
        # En yüksek olasılığı al (Yüzde kaç emin?)
        confidence = max(probability[0]) * 100
        
        st.divider() # Araya çizgi çek
        
        # Sonucu Göster
        if prediction[0] == 0:
            st.error(f"🚨 DİKKAT: Bu haber SAHTE olabilir! (Eminlik Oranı: %{confidence:.2f})")
        else:
            st.success(f"✅ GÜVENİLİR: Bu haber GERÇEK görünüyor. (Eminlik Oranı: %{confidence:.2f})")
            
    else:
        st.warning("Lütfen önce bir metin giriniz.")

# Yan menüye bilgi ekleyelim
st.sidebar.header("Hakkında")

st.sidebar.info("Bu proje, Siyaset Bilimi ve Veri Bilimi kullanılarak geliştirilmiştir. Model, 45.000 haber üzerinde eğitilmiştir.")

