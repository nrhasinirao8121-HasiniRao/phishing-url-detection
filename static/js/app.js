/**
 * app.js
 * ======
 * Client-side interactivity for PhishGuard AI:
 * - Interactive cybersecurity background constellation network with cursor gravity
 * - Asynchronous URL analysis with real-time UI updates
 * - Image modal lightbox for model evaluation plots
 */

document.addEventListener('DOMContentLoaded', () => {
    initCyberCanvas();
});

/* ==========================================================================
   Cybersecurity Background Canvas (Interactive Constellation)
   ========================================================================== */
function initCyberCanvas() {
    const canvas = document.getElementById('cyberCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    let mouse = { x: -1000, y: -1000, radius: 140 };

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    window.addEventListener('mouseleave', () => {
        mouse.x = -1000;
        mouse.y = -1000;
    });

    const particles = [];
    const count = Math.min(75, Math.floor(width / 22));

    for (let i = 0; i < count; i++) {
        const isEmerald = Math.random() > 0.65;
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.45,
            vy: (Math.random() - 0.5) * 0.45,
            radius: Math.random() * 1.8 + 1,
            color: isEmerald ? 'rgba(16, 185, 129, ' : 'rgba(56, 189, 248, '
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Draw connections between nodes
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 130) {
                    const alpha = (1 - dist / 130) * 0.22;
                    ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }

            // Draw line to mouse if nearby
            const dmx = particles[i].x - mouse.x;
            const dmy = particles[i].y - mouse.y;
            const mouseDist = Math.sqrt(dmx * dmx + dmy * dmy);
            if (mouseDist < mouse.radius) {
                const mAlpha = (1 - mouseDist / mouse.radius) * 0.35;
                ctx.strokeStyle = `rgba(16, 185, 129, ${mAlpha})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(mouse.x, mouse.y);
                ctx.stroke();
            }
        }

        // Draw particles
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            ctx.fillStyle = p.color + '0.6)';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();
        }

        requestAnimationFrame(animate);
    }

    animate();
}

/* ==========================================================================
   Quick Fill Helper for Sample URLs
   ========================================================================== */
function fillUrl(url) {
    const input = document.getElementById('urlInput');
    if (input) {
        input.value = url;
        input.focus();
        handleAnalyze(new Event('submit'));
    }
}

/* ==========================================================================
   Asynchronous URL Analysis Handler
   ========================================================================== */
async function handleAnalyze(e) {
    if (e) e.preventDefault();

    const input = document.getElementById('urlInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    const resultCard = document.getElementById('resultCard');

    const urlValue = input.value.trim();
    if (!urlValue) {
        alert('Please enter a URL to analyze.');
        return;
    }

    // Set Loading State
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ url: urlValue })
        });

        const data = await response.json();

        if (!data.success) {
            alert('Analysis Error: ' + (data.error || 'Unknown error occurred.'));
            return;
        }

        renderResult(data);
        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        console.error('Prediction request failed:', err);
        alert('Failed to connect to the prediction server. Please make sure the Flask application is running.');
    } finally {
        // Reset Button State
        btnText.style.display = 'inline-flex';
        btnLoader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

/* ==========================================================================
   Render Prediction Result Card in DOM
   ========================================================================== */
function renderResult(data) {
    const card = document.getElementById('resultCard');
    const isLegit = data.is_legitimate;

    let featuresHtml = '';
    for (const [key, value] of Object.entries(data.features)) {
        featuresHtml += `
            <div class="feature-item">
                <span class="feat-name">${escapeHtml(key)}</span>
                <span class="feat-value">${escapeHtml(String(value))}</span>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="result-header ${data.status_color}">
            <div class="result-badge-container">
                ${isLegit ? `
                    <div class="result-icon legit-icon">
                        <i class="fa-solid fa-shield-check"></i>
                        <span class="badge-ring legit-ring"></span>
                    </div>
                    <div>
                        <h2 class="result-decision legit-text">LEGITIMATE</h2>
                        <span class="result-risk-level safe-tag"><i class="fa-solid fa-check"></i> ${escapeHtml(data.risk_level)}</span>
                    </div>
                ` : `
                    <div class="result-icon phish-icon">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <span class="badge-ring phish-ring"></span>
                    </div>
                    <div>
                        <h2 class="result-decision phish-text">PHISHING DETECTED</h2>
                        <span class="result-risk-level danger-tag"><i class="fa-solid fa-skull-crossbones"></i> ${escapeHtml(data.risk_level)}</span>
                    </div>
                `}
            </div>
            
            <div class="confidence-box">
                <span class="confidence-title">Confidence Rating</span>
                <span class="confidence-value">${data.confidence}%</span>
            </div>
        </div>

        <div class="result-url-display">
            <span class="url-label">Analyzed Target URL:</span>
            <code class="url-code" id="analyzedUrlDisplay">${escapeHtml(data.url)}</code>
        </div>

        <div class="prob-distribution">
            <div class="prob-labels">
                <span class="prob-label phish-label"><i class="fa-solid fa-bug"></i> Phishing Probability: <strong>${data.prob_phishing}%</strong></span>
                <span class="prob-label legit-label"><i class="fa-solid fa-shield-heart"></i> Legitimate Probability: <strong>${data.prob_legitimate}%</strong></span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar phish-bar" style="width: ${data.prob_phishing}%;"></div>
                <div class="prob-bar legit-bar" style="width: ${data.prob_legitimate}%;"></div>
            </div>
        </div>

        <div class="features-breakdown">
            <h3 class="breakdown-title"><i class="fa-solid fa-fingerprint"></i> Deep URL Feature Telemetry</h3>
            <div class="feature-grid">
                ${featuresHtml}
            </div>
        </div>
    `;
}

function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

/* ==========================================================================
   Image Modal Lightbox
   ========================================================================== */
function openModal(imgSrc, caption) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalCaption = document.getElementById('modalCaption');

    modalImg.src = imgSrc;
    modalCaption.textContent = caption;
    modal.style.display = 'flex';
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    if (modal) modal.style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});
