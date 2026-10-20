import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from src.models.pipeline import MLServingPipeline
from src.config import PRESENTATION_DIR

# 1. Page Configuration
st.set_page_config(
    page_title="Practical AI in Industry | Hotel Ranking & Cross-Sell",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background-color: #07090e;
        color: #f1f5f9;
    }
    
    .metric-card {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(12px);
        margin-bottom: 12px;
    }
    
    .hotel-card-st {
        background: rgba(23, 31, 48, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hotel-card-st:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .ml-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(56, 189, 248, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .price-text {
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        font-size: 1.35rem;
        font-weight: 800;
    }
    
    .tag-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cache Singleton Pipeline
@st.cache_resource(show_spinner="Initializing Machine Learning Models & Catalog...")
def load_pipeline():
    pipeline = MLServingPipeline.get_instance()
    if not pipeline.is_loaded:
        pipeline.load()
    return pipeline

pipeline = load_pipeline()

# 4. Helper formatters
def format_idr(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

# 5. Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hotel-building.png", width=64)
    st.title("Search Controls")
    st.caption("Synchronous ML Ranking Parameters")
    
    city = st.selectbox(
        "📍 Destination City",
        options=["Bali", "Jakarta", "Bandung", "Yogyakarta", "Surabaya", "Lombok", "Labuan Bajo", "All"],
        index=0
    )
    
    budget = st.slider(
        "💰 Target Budget / Night",
        min_value=400_000,
        max_value=4_000_000,
        value=1_500_000,
        step=100_000,
        format="Rp %d"
    )
    st.write(f"Selected Budget: **{format_idr(budget)}**")
    
    pref_stars = st.radio(
        "⭐ Preferred Star Rating",
        options=[3, 4, 5],
        index=1,
        horizontal=True
    )
    
    strategy = st.radio(
        "⚡ Ranking Algorithm",
        options=["ML Conversion AI", "Lowest Price", "Top Rated"],
        index=0
    )
    strategy_code = {
        "ML Conversion AI": "ml_ranking",
        "Lowest Price": "price_asc",
        "Top Rated": "rating_desc"
    }[strategy]
    
    st.divider()
    st.markdown("### 👤 Speaker & Inspiration")
    st.info("**Setia Budi**\nMachine Learning Engineer at **tiket.com**\n*Marp Keynote Deck Included*")

# 6. Main Header & Telemetry Metrics
st.title("🏨 Practical AI in Industry: Hotel Search Ranking")
st.markdown(
    "Production-grade Machine Learning engine demonstrating **Pointwise Search Ranking** "
    "and **Bayesian Cross-Sell Recommendations** inspired by Online Travel Agent (OTA) architecture."
)

# Execute Inference
t0 = time.time()
result = pipeline.rank_hotels(
    city=city,
    user_budget=budget,
    pref_stars=pref_stars,
    sort_by=strategy_code,
    limit=18
)
exec_time_ms = round((time.time() - t0) * 1000, 2)

# Metrics Banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Candidates Scored", f"{result['total']} Hotels")
with col_m2:
    st.metric("Server Latency", f"{exec_time_ms} ms", delta="Sub-15ms Target", delta_color="normal")
with col_m3:
    st.metric("Model ROC-AUC", "0.7459", delta="+0.808 PR-AUC")
with col_m4:
    st.metric("Active Strategy", strategy)

# 7. Main Tabs
tab_ranking, tab_slides, tab_arch = st.tabs([
    "🏨 Live Ranked Accommodations",
    "📊 Keynote Slide Deck",
    "🏛 System Architecture & MLOps"
])

# TAB 1: RANKED HOTELS
with tab_ranking:
    st.subheader(f"Ranked Results for {city} (Budget: {format_idr(budget)})")
    
    hotels = result.get("hotels", [])
    if not hotels:
        st.warning("No accommodations match the selected criteria.")
    else:
        # Display in 2-column responsive layout
        for i in range(0, len(hotels), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(hotels):
                    h = hotels[i + j]
                    with cols[j]:
                        with st.container():
                            # Header with ML Score
                            ml_score_badge = ""
                            if "ml_score" in h:
                                ml_score_badge = f"<span class='ml-badge'>⚡ {h['ml_score']}% Match</span>"
                            
                            st.markdown(f"""
                            <div class="hotel-card-st">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div>
                                        <h3 style="margin: 0; color: #fff; font-size: 1.15rem;">{h['name']}</h3>
                                        <p style="color: #94a3b8; font-size: 0.8rem; margin: 2px 0 8px 0;">
                                            📍 {h['city']} • {h['distance_to_center_km']} km to center
                                        </p>
                                    </div>
                                    <div>{ml_score_badge}</div>
                                </div>
                                <div style="margin-bottom: 8px;">
                                    <span style="color: #f59e0b; font-weight: 700;">{'★' * h['star_rating']}</span>
                                    <span style="color: #10b981; font-weight: 700; margin-left: 6px;">{h['review_score']}</span>
                                    <span style="color: #64748b; font-size: 0.75rem;">({h['review_count']} reviews)</span>
                                </div>
                                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">{h['description']}</p>
                                <div style="margin: 10px 0;">
                                    {''.join([f"<span class='tag-badge'>{b}</span>" for b in h.get('badges', [])])}
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px; margin-top: 10px;">
                                    <div>
                                        <div class="price-text">{format_idr(h['price_per_night'])}</div>
                                        <span style="color: #64748b; font-size: 0.7rem;">per night • incl. taxes</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Expandable Cross-Sell Add-ons
                            with st.expander(f"🎯 Recommended Add-ons for {h['name']}"):
                                cross_sells = pipeline.get_cross_sell(h["hotel_id"], top_k=3)
                                recs = cross_sells.get("recommendations", [])
                                if recs:
                                    st.caption("Pilihan aktivitas/layanan yang paling sering dipesan bersama properti ini:")
                                    for item in recs:
                                        c_col1, c_col2 = st.columns([3, 1])
                                        with c_col1:
                                            st.markdown(f"**{item['name']}** ({item['category']})")
                                            st.caption(f"★ {item['confidence_percent']}% of guests also added this")
                                        with c_col2:
                                            st.markdown(f"**+{format_idr(item['price'])}**")
                                            st.checkbox("Add", key=f"cs_{h['hotel_id']}_{item['item_id']}")
                                else:
                                    st.write("No specific add-on recommendations available.")

# TAB 2: SLIDE PRESENTASI
with tab_slides:
    st.subheader("Keynote Presentation: A Brief Story of AI Implementation in Industry")
    st.caption("Oleh Setia Budi (Machine Learning Engineer di tiket.com)")
    
    slide_path = PRESENTATION_DIR / "presentation.html"
    if slide_path.exists():
        with open(slide_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.info("💡 **Tips Presentasi:** Klik slide di bawah, lalu tekan tombol **P** untuk masuk ke Presenter Mode dengan speaker notes dan timer, atau tekan **F** untuk Fullscreen.")
        components.html(html_content, height=650, scrolling=True)
    else:
        st.error("Slide presentation.html tidak ditemukan di folder presentation/.")

# TAB 3: SYSTEM ARCHITECTURE & MLOPS
with tab_arch:
    st.subheader("Arsitektur Sistem & Perbandingan Industri")
    
    st.markdown("""
    ### 🎯 Mengapa Arsitektur Ini Dibuat?
    Seperti dijelaskan dalam presentasi, **model hanyalah 10–20% dari pekerjaan**. Sekitar **80% pekerjaan ML Engineer adalah rekayasa sistem (engineering)**:
    - **Data Pipeline:** Memproses data mentah menjadi fitur yang bermakna.
    - **Inference Latency:** Menjaga respon inferensi di bawah **15–50 milidetik** agar tidak memperlambat pengalaman pengguna di aplikasi.
    - **Model Deployment:** Mengemas model ke dalam service yang siap diakses 24/7.
    
    ### 📊 Fitur-Fitur Model Ranking (14 Dimensi):
    Model `HistGradientBoostingClassifier` dilatih menggunakan kombinasi fitur bisnis:
    1. **Price Match:** `price_ratio` & `price_gaussian` (kurva utilitas kesediaan membayar).
    2. **Star Preference:** `star_exact_match` dan selisih bintang (`star_diff`).
    3. **Quality & Popularity:** `review_score`, `rating_scaled`, dan `log(1 + review_count)`.
    4. **Location Decay:** `distance_to_center_km` dan pembobotan eksponensial `exp(-distance / 10)`.
    5. **Promotion Incentive:** Persentase dan faktor diskon.
    6. **Value Index:** Rasio skor ulasan terhadap harga kamar.
    
    ### 🔄 Perbedaan FastAPI vs Streamlit di Industri:
    - **FastAPI:** Digunakan untuk *production microservice high-concurrency* yang dihubungkan ke aplikasi mobile (tiket.com Android/iOS) dan frontend web.
    - **Streamlit:** Digunakan oleh tim Data/AI untuk *rapid prototyping*, *dashboard evaluasi internal*, dan *demo portofolio publik*.
    """)
