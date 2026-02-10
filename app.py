import streamlit as st
import joblib

# 1. Kaydedilen Modelleri Geri Yükle (Cache kullanarak hızlandırıyoruz)
@st.cache_resource
def model_yukle():
    # Dosya isimlerinin senin kaydettiklerinle AYNI olduğundan emin ol
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('random_forest_model.pkl')
    return vectorizer, model

# Hata alırsak kullanıcıya göstermek için try-except bloğu
try:
    cv, rf_model = model_yukle()
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