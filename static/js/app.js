/**
 * CardioAI - Disease Prediction from Medical Data
 * Frontend Interaction and API Client
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSliders();
    initPresets();
    initPredictionForm();
});

// Tab Navigation
function initTabs() {
    const tabButtons = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(`tab-${targetTab}`);
            if (targetEl) targetEl.classList.add('active');
        });
    });
}

// Range Sliders with live value badges
function initSliders() {
    const sliders = [
        { inputId: 'age', badgeId: 'valAge' },
        { inputId: 'trestbps', badgeId: 'valTrestbps' },
        { inputId: 'chol', badgeId: 'valChol' },
        { inputId: 'thalach', badgeId: 'valThalach' },
        { inputId: 'oldpeak', badgeId: 'valOldpeak' }
    ];

    sliders.forEach(({ inputId, badgeId }) => {
        const input = document.getElementById(inputId);
        const badge = document.getElementById(badgeId);
        if (input && badge) {
            input.addEventListener('input', () => {
                badge.textContent = input.value;
            });
        }
    });
}

// Preset Profiles
function initPresets() {
    const presets = {
        high: {
            age: 67, sex: '1', cp: '4', trestbps: 160, chol: 286,
            fbs: '0', restecg: '2', thalach: 108, exang: '1',
            oldpeak: 1.5, slope: '2', ca: '3', thal: '3'
        },
        low: {
            age: 41, sex: '0', cp: '2', trestbps: 120, chol: 180,
            fbs: '0', restecg: '0', thalach: 175, exang: '0',
            oldpeak: 0.0, slope: '1', ca: '0', thal: '3'
        },
        borderline: {
            age: 55, sex: '1', cp: '3', trestbps: 135, chol: 245,
            fbs: '1', restecg: '1', thalach: 140, exang: '0',
            oldpeak: 1.0, slope: '2', ca: '1', thal: '6'
        },
        reset: {
            age: 55, sex: '1', cp: '4', trestbps: 130, chol: 230,
            fbs: '0', restecg: '0', thalach: 150, exang: '0',
            oldpeak: 1.0, slope: '2', ca: '0', thal: '3'
        }
    };

    function applyPreset(data) {
        for (const [key, val] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = val;
                // Dispatch input event to refresh badges
                el.dispatchEvent(new Event('input'));
            }
        }
    }

    const btnHigh = document.getElementById('btnPresetHigh');
    const btnLow = document.getElementById('btnPresetLow');
    const btnBorderline = document.getElementById('btnPresetBorderline');
    const btnReset = document.getElementById('btnReset');

    if (btnHigh) btnHigh.addEventListener('click', () => {
        applyPreset(presets.high);
        submitPrediction();
    });

    if (btnLow) btnLow.addEventListener('click', () => {
        applyPreset(presets.low);
        submitPrediction();
    });

    if (btnBorderline) btnBorderline.addEventListener('click', () => {
        applyPreset(presets.borderline);
        submitPrediction();
    });

    if (btnReset) btnReset.addEventListener('click', () => {
        applyPreset(presets.reset);
    });
}

// Prediction Form Handling
function initPredictionForm() {
    const form = document.getElementById('patientForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            submitPrediction();
        });
    }
}

async function submitPrediction() {
    const form = document.getElementById('patientForm');
    const submitBtn = document.getElementById('btnSubmitPredict');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');

    const formData = new FormData(form);
    const payload = {};
    for (const [key, val] of formData.entries()) {
        payload[key] = val;
    }

    // Set loading state
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Server error during prediction');
        }

        const data = await response.json();
        renderPredictionResults(data);

    } catch (err) {
        alert(`Prediction Error: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        btnText.style.display = 'inline-flex';
        btnLoader.style.display = 'none';
    }
}

function renderPredictionResults(data) {
    const resultsCard = document.getElementById('resultsCard');
    const placeholder = resultsCard.querySelector('.placeholder-content');
    const activeResult = document.getElementById('activeResult');

    if (placeholder) placeholder.style.display = 'none';
    resultsCard.classList.remove('empty-state');
    activeResult.style.display = 'block';

    const primary = data.primary;
    const isPresent = primary.status === 'Present';

    // Model & Timestamp Badges
    document.getElementById('resModelBadge').textContent = primary.name;
    document.getElementById('resTimestamp').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Diagnosis Banner
    const banner = document.getElementById('diagnosisBanner');
    const bannerIcon = document.getElementById('bannerIcon');
    const diseaseClass = document.getElementById('resDiseaseClass');
    const subStatus = document.getElementById('resSubStatus');

    if (isPresent) {
        banner.className = 'diagnosis-banner present';
        bannerIcon.textContent = '⚠️';
        diseaseClass.textContent = 'DISEASE PRESENT';
        subStatus.textContent = `High statistical probability (${primary.probability}%) of coronary heart disease.`;
    } else {
        banner.className = 'diagnosis-banner absent';
        bannerIcon.textContent = '🛡️';
        diseaseClass.textContent = 'DISEASE ABSENT';
        subStatus.textContent = `Low statistical likelihood (${primary.probability}%) based on clinical parameters.`;
    }

    // Meter Bar
    const probText = document.getElementById('resProbText');
    const probBar = document.getElementById('resProbBar');
    probText.textContent = `${primary.probability}%`;
    probBar.style.width = `${Math.min(Math.max(primary.probability, 4), 100)}%`;

    // Multi-Algorithm Breakdown Grid
    const algoGrid = document.getElementById('algoCardsGrid');
    algoGrid.innerHTML = '';
    for (const [key, algo] of Object.entries(data.all_models)) {
        const card = document.createElement('div');
        card.className = 'algo-card';
        const isAlgoPresent = algo.status === 'Present';
        card.innerHTML = `
            <span class="name">${algo.name}</span>
            <span class="status-pill ${isAlgoPresent ? 'present' : 'absent'}">
                ${algo.status} (${algo.probability}%)
            </span>
        `;
        algoGrid.appendChild(card);
    }

    // Risk Factors List
    const riskList = document.getElementById('riskFactorsList');
    riskList.innerHTML = '';
    if (data.risk_factors && data.risk_factors.length > 0) {
        riskList.className = 'risk-list';
        data.risk_factors.forEach(rf => {
            const li = document.createElement('li');
            li.textContent = `[!] ${rf}`;
            riskList.appendChild(li);
        });
    } else {
        riskList.className = 'risk-list clear';
        const li = document.createElement('li');
        li.textContent = '✓ No high-risk anomalies flagged in entered clinical thresholds.';
        riskList.appendChild(li);
    }
}
