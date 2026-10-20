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

# 2. Comprehensive CSS Styling for Unified Dark Glassmorphism UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap');
    
    /* Global Base */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #07090e !important;
        color: #f1f5f9 !important;
    }
    
    /* Force Sidebar Dark Mode */
    [data-testid="stSidebar"] {
        background-color: #0c101a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(14, 19, 31, 0.8), rgba(23, 31, 48, 0.6));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }
    .hero-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 8px;
        margin-bottom: 10px;
    }
    .hero-badge.purple {
        background: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        border-color: rgba(129, 140, 248, 0.3);
    }
    
    /* Telemetry Grid */
    .telemetry-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    @media (max-width: 768px) {
        .telemetry-bar { grid-template-columns: repeat(2, 1fr); }
    }
    .telem-box {
        background: rgba(18, 24, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .telem-box:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }
    .telem-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .telem-value {
        font-size: 1.25rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #fff;
    }
    .telem-value.cyan { color: #38bdf8; }
    .telem-value.emerald { color: #10b981; }
    
    /* Hotel Card with Image Cover */
    .hotel-card-st {
        background: rgba(23, 31, 48, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 16px;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .hotel-card-st:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4), 0 0 15px rgba(56, 189, 248, 0.15);
    }
    
    .hotel-image-header {
        position: relative;
        height: 190px;
        width: 100%;
        overflow: hidden;
        background: #0f1422;
    }
    .hotel-image-header img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .hotel-card-st:hover .hotel-image-header img {
        transform: scale(1.06);
    }
    .img-gradient-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(7, 9, 14, 0.2) 0%, rgba(7, 9, 14, 0.85) 100%);
    }
    .img-badges-float {
        position: absolute;
        top: 12px;
        left: 12px;
        right: 12px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        z-index: 2;
    }
    
    .hotel-body-inner {
        padding: 16px 20px 20px;
    }
    
    .ml-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(56, 189, 248, 0.25));
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34d399;
        font-weight: 800;
        font-size: 0.82rem;
        padding: 5px 10px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        backdrop-filter: blur(8px);
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
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-right: 6px;
    }
    .tag-badge.deal {
        background: rgba(244, 63, 94, 0.15);
        border-color: rgba(244, 63, 94, 0.3);
        color: #fb7185;
    }
    .tag-badge.fav {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.3);
        color: #fbbf24;
    }
    
    /* Streamlit Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(18, 24, 38, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
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
    st.markdown("## 🏨 OTA Search Filters")
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
    st.write(f"Target Spend: **{format_idr(budget)}**")
    
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
    st.info("**Setia Budi**\nMachine Learning Engineer at **tiket.com**\n*Keynote Presentation Included*")

# 6. Main Header Hero Banner
st.markdown("""
<div class="hero-container">
    <div>
        <span class="hero-badge">tiket.com Inspired Architecture</span>
        <span class="hero-badge purple">Synchronous ML Serving</span>
    </div>
    <h1 style="color: #fff; font-size: 2rem; font-weight: 800; margin: 4px 0 10px 0; letter-spacing: -0.02em;">
        OTA Production AI: Hotel Search Ranking & Recommender
    </h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0; max-width: 800px;">
        Production-grade Machine Learning engine demonstrating <b>Pointwise Search Ranking</b> (HistGradientBoosting) 
        and <b>Bayesian Cross-Sell Recommendations</b> with sub-15ms inference latency.
    </p>
</div>
""", unsafe_allow_html=True)

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

# Custom Telemetry Bar
st.markdown(f"""
<div class="telemetry-bar">
    <div class="telem-box">
        <span class="telem-label">Candidates Scored</span>
        <span class="telem-value">{result['total']} Hotels</span>
    </div>
    <div class="telem-box">
        <span class="telem-label">Server Latency</span>
        <span class="telem-value cyan">{exec_time_ms} ms</span>
    </div>
    <div class="telem-box">
        <span class="telem-label">Model ROC-AUC</span>
        <span class="telem-value emerald">0.7459</span>
    </div>
    <div class="telem-box">
        <span class="telem-label">Ranking Mode</span>
        <span class="telem-value cyan">{strategy}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Main Tabs
tab_ranking, tab_slides, tab_arch = st.tabs([
    "🏨 Live Ranked Accommodations",
    "📊 Keynote Slide Deck",
    "🏛 System Architecture & MLOps"
])

# TAB 1: RANKED HOTELS
with tab_ranking:
    st.markdown(f"### Ranked Accommodations in **{city}** (Budget Target: {format_idr(budget)})")
    
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
                            # Badges
                            badges_html = ""
                            for b in h.get("badges", []):
                                cls = "tag-badge"
                                if "Deal" in b: cls += " deal"
                                if "Favorite" in b: cls += " fav"
                                badges_html += f"<span class='{cls}'>{b}</span>"
                                
                            ml_score_badge = ""
                            if "ml_score" in h:
                                ml_score_badge = f"<span class='ml-badge'>⚡ {h['ml_score']}% Match</span>"
                                
                            img_url = h.get("image_url") or "https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=800&q=80"
                            
                            st.markdown(f"""
                            <div class="hotel-card-st">
                                <div class="hotel-image-header">
                                    <img src="{img_url}" alt="{h['name']}" loading="lazy">
                                    <div class="img-gradient-overlay"></div>
                                    <div class="img-badges-float">
                                        <div style="display: flex; gap: 4px; flex-wrap: wrap;">{badges_html}</div>
                                        <div>{ml_score_badge}</div>
                                    </div>
                                </div>
                                <div class="hotel-body-inner">
                                    <h3 style="margin: 0 0 4px 0; color: #fff; font-size: 1.15rem;">{h['name']}</h3>
                                    <p style="color: #94a3b8; font-size: 0.8rem; margin: 0 0 10px 0;">
                                        📍 {h['city']} • {h['distance_to_center_km']} km to center
                                    </p>
                                    <div style="margin-bottom: 8px;">
                                        <span style="color: #f59e0b; font-weight: 700;">{'★' * h['star_rating']}</span>
                                        <span style="color: #10b981; font-weight: 700; margin-left: 6px;">{h['review_score']}</span>
                                        <span style="color: #64748b; font-size: 0.75rem;">({h['review_count']} reviews)</span>
                                    </div>
                                    <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4; margin-bottom: 12px;">{h['description']}</p>
                                    <div style="display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                                        <div>
                                            <div class="price-text">{format_idr(h['price_per_night'])}</div>
                                            <span style="color: #64748b; font-size: 0.7rem;">per night • incl. taxes</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Expandable Cross-Sell Add-ons
                            with st.expander(f"🎯 Add-ons & Activities for {h['name']}"):
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
        st.info("💡 **Tips Presentasi:** Klik area slide di bawah, lalu tekan tombol **P** untuk masuk ke Presenter Mode dengan speaker notes dan timer, atau tekan **F** untuk Fullscreen.")
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
