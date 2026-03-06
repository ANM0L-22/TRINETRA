document.addEventListener('DOMContentLoaded', () => {
  // Sidebar toggle (for desktop collapse and mobile show/hide)
  const sidebar = document.getElementById('sidebar');
  const sidebarToggle = document.getElementById('sidebarToggle');
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      document.querySelector('.main')?.classList.toggle('shifted');
    });
  }

  // Theme toggle (light/dark)
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('light');
      // store preference
      if (document.body.classList.contains('light')) localStorage.setItem('theme','light'); else localStorage.removeItem('theme');
      // animate icon
      themeToggle.classList.add('rotating');
      setTimeout(() => themeToggle.classList.remove('rotating'), 400);
      // update chart colors if already created
      if (trafficChart) {
        const isLight = document.body.classList.contains('light');
        trafficChart.options.scales.x.ticks.color = isLight ? '#0b1220' : '#E6EDF3';
        trafficChart.options.scales.y.ticks.color = isLight ? '#0b1220' : '#E6EDF3';
        trafficChart.update();
      }
    });
    // apply stored preference
    if (localStorage.getItem('theme') === 'light') document.body.classList.add('light');
    // if chart is already created later, update color when loaded
  }

  // Animated counters
  const counters = document.querySelectorAll('.counter');
  // Function to animate a single counter
  const animateCounter = (el, target, duration = 1200) => {
    if (!el) return;
    el.textContent = '0';
    const start = 0;
    const range = target - start;
    const startTime = performance.now();
    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const val = Math.floor(start + range * progress);
      el.textContent = val.toString();
      if (progress < 1) requestAnimationFrame(step); else el.textContent = target.toString();
    };
    requestAnimationFrame(step);
  };

  // Chart.js: initialize safely
  const ctx = document.getElementById('trafficChart');
  // Chart creation function
  let trafficChart = null;
  const createChart = (labels, dataPoints) => {
    if (!(ctx && window.Chart)) return;
    const cfg = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Vehicles',
          data: dataPoints,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#6C4DF6',
          backgroundColor: 'rgba(108,77,246,0.14)',
          tension: 0.28,
          fill: true,
          pointRadius: 3
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: document.body.classList.contains('light') ? '#0b1220' : '#E6EDF3' }, grid: { color: 'transparent' } },
          y: { ticks: { color: document.body.classList.contains('light') ? '#0b1220' : '#E6EDF3' }, grid: { color: 'rgba(255,255,255,0.03)' } }
        }
      }
    });
    trafficChart = cfg;
    return cfg;
  };

  // Fetch local data.json and populate UI
  const dataUrl = 'data.json';
  fetch(dataUrl).then(r => {
    if (!r.ok) throw new Error('Failed to load data.json');
    return r.json();
  }).then(data => {
    // update counters
    const map = [
      {sel: '.counter[data-target="1245"]', key: 'vehiclesDetected'},
      {sel: '.counter[data-target="89"]', key: 'violationsToday'}
    ];
    map.forEach(m => {
      const el = document.querySelector(m.sel);
      const val = data[m.key] ?? 0;
      animateCounter(el, val);
    });

    // update other stat elements
    const densityEl = Array.from(document.querySelectorAll('.card p')).find(p => p.textContent.includes('%'));
    if (densityEl) densityEl.textContent = (data.trafficDensity ?? 0) + '%';
    const camsEl = Array.from(document.querySelectorAll('.card p')).find(p => p.textContent.trim() === '16');
    if (camsEl) camsEl.textContent = (data.activeCameras ?? 0).toString();

    // chart
    const labels = data.hourly?.labels ?? [];
    const points = data.hourly?.data ?? [];
    createChart(labels.length? labels : ['6AM','9AM','12PM','3PM','6PM'], points.length? points : [80,120,180,240,200]);

    // render cameras
    const camerasWrap = document.getElementById('cameras');
    if (camerasWrap && Array.isArray(data.cameras)) {
      camerasWrap.innerHTML = '';
      data.cameras.forEach(cam => {
        const div = document.createElement('div');
        div.className = 'camera-card';
        div.innerHTML = `
          <img class="cam-thumb" src="${cam.thumbnail}" alt="${cam.name} thumbnail">
          <div class="cam-info">
            <h4>${cam.name}</h4>
            <div class="cam-meta">
              <div class="badge ${cam.status==='online'?'online':'offline'}">${cam.status}</div>
              <div class="muted">Violations: ${cam.violations}</div>
              <div class="muted">Last: ${cam.lastSeen}</div>
            </div>
          </div>
          <div class="cam-actions">
            <button class="btn" onclick="alert('Open live for ${cam.name}')">View</button>
            <button class="btn" onclick="alert('Open details for ${cam.name}')">Details</button>
          </div>
        `;
        camerasWrap.appendChild(div);
      });
    }
  }).catch(err => {
    console.warn('Data load failed:', err);
    // fallback: create default chart
    createChart(['6AM','9AM','12PM','3PM','6PM'], [80,120,180,240,200]);
  });
});