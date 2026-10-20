// State
let currentQuery = {
  city: "Bali",
  user_budget: 1500000,
  pref_stars: 4,
  sort_by: "ml_ranking",
  limit: 18
};

// Elements
const citySelect = document.getElementById("city-select");
const budgetSlider = document.getElementById("budget-slider");
const budgetDisplay = document.getElementById("budget-display");
const starButtons = document.querySelectorAll(".star-btn");
const strategyButtons = document.querySelectorAll(".strategy-btn");
const refreshBtn = document.getElementById("refresh-btn");
const hotelsContainer = document.getElementById("hotels-container");
const resultsMetaText = document.getElementById("results-meta-text");

// Telemetry Elements
const telemLatency = document.getElementById("telem-latency");
const telemAuc = document.getElementById("telem-auc");
const telemTotal = document.getElementById("telem-total");
const telemStrategy = document.getElementById("telem-strategy");
const headerLatencyText = document.getElementById("header-latency-text");
const modelNameText = document.getElementById("model-name-text");
const healthStatusText = document.getElementById("health-status-text");

// Modal Elements
const modalOverlay = document.getElementById("cross-sell-modal");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalCancelBtn = document.getElementById("modal-cancel-btn");
const modalConfirmBtn = document.getElementById("modal-confirm-btn");
const modalHotelTitle = document.getElementById("modal-hotel-title");
const crossSellContainer = document.getElementById("cross-sell-container");

// Format IDR Currency
function formatIDR(amount) {
  return "Rp " + new Intl.NumberFormat("id-ID").format(amount);
}

// Fetch System Health
async function fetchHealth() {
  try {
    const res = await fetch("/api/v1/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    
    if (data.ml_models_ready) {
      healthStatusText.textContent = "FastAPI: Live";
      if (data.model_details && data.model_details.metrics) {
        telemAuc.textContent = data.model_details.metrics.roc_auc.toFixed(3);
      }
      if (data.model_details && data.model_details.model_type) {
        modelNameText.textContent = "GradBoosting";
      }
    }
  } catch (err) {
    console.error("Health check error:", err);
    healthStatusText.textContent = "Offline";
  }
}

// Perform Hotel Search & ML Ranking
async function searchHotels() {
  hotelsContainer.innerHTML = `
    <div style="grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #94a3b8;">
      <div class="pulse-dot" style="margin: 0 auto 16px; width: 12px; height: 12px;"></div>
      <p>Running ML Pointwise Conversion Ranker...</p>
    </div>
  `;

  try {
    const t0 = performance.now();
    const res = await fetch("/api/v1/hotels/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentQuery)
    });
    
    if (!res.ok) throw new Error("Search request failed");
    const data = await res.json();
    const clientLatency = Math.round(performance.now() - t0);
    
    // Update Telemetry
    telemLatency.textContent = `${data.latency_ms} ms`;
    headerLatencyText.textContent = `${data.latency_ms} ms`;
    telemTotal.textContent = data.total;
    
    const strategyLabels = {
      "ml_ranking": "ML Conversion AI",
      "price_asc": "Lowest Price",
      "rating_desc": "Top Rated"
    };
    telemStrategy.textContent = strategyLabels[data.sort_applied] || data.sort_applied;
    
    resultsMetaText.textContent = `Found ${data.total} accommodations in ${data.latency_ms}ms server time (round-trip ${clientLatency}ms)`;

    renderHotels(data.hotels);
  } catch (err) {
    console.error("Search failed:", err);
    hotelsContainer.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: #f43f5e;">
        <p>⚠️ Failed to fetch ranked accommodations. Ensure server is running.</p>
      </div>
    `;
  }
}

// Render Hotels Cards
function renderHotels(hotels) {
  if (!hotels || hotels.length === 0) {
    hotelsContainer.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 60px; color: #64748b;">
        <p>No accommodations found matching the current criteria.</p>
      </div>
    `;
    return;
  }

  hotelsContainer.innerHTML = hotels.map(hotel => {
    const hasDiscount = hotel.discount_percent > 0;
    const originalPrice = hasDiscount 
      ? Math.round(hotel.price_per_night / (1 - (hotel.discount_percent / 100)))
      : null;

    // Badges HTML
    const badgesHtml = (hotel.badges || []).map(b => {
      let cls = "meta-badge";
      if (b.includes("Deal")) cls += " highlight-deal";
      if (b.includes("Favorite")) cls += " highlight-fav";
      return `<span class="${cls}">${b}</span>`;
    }).join("");

    // Star string
    const stars = "★".repeat(hotel.star_rating);

    // ML Score badge if available
    const mlScoreHtml = (hotel.ml_score !== undefined && hotel.ml_score !== null)
      ? `
        <div class="ml-score-badge" title="Estimated Conversion Probability">
          <span class="ml-score-val">${hotel.ml_score}%</span>
          <span class="ml-score-label">ML Match</span>
        </div>
      `
      : "";

    return `
      <div class="hotel-card" data-hotel-id="${hotel.hotel_id}">
        <div>
          <div class="hotel-card-top">
            <div>
              <h4 class="hotel-name">${hotel.name}</h4>
              <span class="hotel-city-badge">📍 ${hotel.city} • ${hotel.distance_to_center_km} km to center</span>
            </div>
            ${mlScoreHtml}
          </div>

          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 0.85rem;">
            <span style="color: #f59e0b; font-weight: 700;">${stars}</span>
            <span style="color: #10b981; font-weight: 700;">${hotel.review_score}</span>
            <span style="color: #64748b; font-size: 0.75rem;">(${hotel.review_count} reviews)</span>
          </div>

          <p class="hotel-desc">${hotel.description}</p>

          <div class="badges-container">
            ${badgesHtml}
          </div>
        </div>

        <div class="card-bottom">
          <div>
            ${hasDiscount ? `<span style="font-size: 0.75rem; color: #64748b; text-decoration: line-through;">${formatIDR(originalPrice)}</span>` : ""}
            <div class="hotel-price">${formatIDR(hotel.price_per_night)}</div>
            <span class="hotel-price-unit">per night • incl. taxes</span>
          </div>

          <button class="book-btn" onclick="openCrossSellModal('${hotel.hotel_id}', '${encodeURIComponent(hotel.name)}')">
            <span>Book & Add-ons</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>
      </div>
    `;
  }).join("");
}

// Open Cross-Sell Recommendation Modal
async function openCrossSellModal(hotelId, encodedName) {
  const hotelName = decodeURIComponent(encodedName);
  modalHotelTitle.textContent = hotelName;
  crossSellContainer.innerHTML = `
    <div style="text-align: center; padding: 30px; color: #94a3b8;">
      <p>Running real-time cross-sell recommender for ${hotelName}...</p>
    </div>
  `;
  modalOverlay.classList.add("open");

  try {
    const res = await fetch(`/api/v1/recommend/cross-sell?hotel_id=${hotelId}&top_k=4`);
    if (!res.ok) throw new Error("Cross-sell fetch failed");
    const data = await res.json();
    
    renderCrossSells(data.recommendations);
  } catch (err) {
    console.error("Cross-sell fetch error:", err);
    crossSellContainer.innerHTML = `
      <p style="color: #f43f5e; padding: 20px; text-align: center;">Unable to load recommendations.</p>
    `;
  }
}

// Render Cross-Sell Items
function renderCrossSells(items) {
  if (!items || items.length === 0) {
    crossSellContainer.innerHTML = `<p style="color: #64748b; text-align: center;">No specific add-ons found.</p>`;
    return;
  }

  crossSellContainer.innerHTML = items.map(item => `
    <div class="cs-item-card">
      <div class="cs-item-info">
        <span class="cs-category-badge">${item.category}</span>
        <span class="cs-item-name">${item.name}</span>
        <span class="cs-confidence">★ ${item.confidence_percent}% of guests also added this</span>
      </div>
      <div class="cs-item-price-side">
        <span class="cs-price">+${formatIDR(item.price)}</span>
        <input type="checkbox" class="add-addon-checkbox" title="Add to booking">
      </div>
    </div>
  `).join("");
}

// Event Listeners
function setupEvents() {
  // City Select
  citySelect.addEventListener("change", (e) => {
    currentQuery.city = e.target.value;
    searchHotels();
  });

  // Budget Slider
  let budgetTimeout;
  budgetSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    budgetDisplay.textContent = formatIDR(val);
    currentQuery.user_budget = val;
    clearTimeout(budgetTimeout);
    budgetTimeout = setTimeout(searchHotels, 150);
  });

  // Star Buttons
  starButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      starButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentQuery.pref_stars = parseInt(btn.dataset.star);
      searchHotels();
    });
  });

  // Strategy Buttons
  strategyButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      strategyButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentQuery.sort_by = btn.dataset.sort;
      searchHotels();
    });
  });

  // Refresh
  refreshBtn.addEventListener("click", searchHotels);

  // Modal Close
  modalCloseBtn.addEventListener("click", () => modalOverlay.classList.remove("open"));
  modalCancelBtn.addEventListener("click", () => modalOverlay.classList.remove("open"));
  modalConfirmBtn.addEventListener("click", () => {
    alert("Reservation & selected add-ons confirmed! (Simulation completed)");
    modalOverlay.classList.remove("open");
  });
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) modalOverlay.classList.remove("open");
  });
}

// Init
window.addEventListener("DOMContentLoaded", () => {
  setupEvents();
  fetchHealth();
  searchHotels();
});
