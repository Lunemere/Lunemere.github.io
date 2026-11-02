/* 花哨效果：飘落花瓣、卡片 3D 倾斜 + 光泽 + 滚动入场、打字机签名、鼠标点击/拖尾特效。
   系统开启「减少动态效果」时全部关闭。 */
(function () {
  "use strict";
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  var isDark = function () { return document.documentElement.classList.contains("dark"); };

  /* ---------- 1. 飘落花瓣 ---------- */
  (function petals() {
    var cv = document.createElement("canvas");
    cv.id = "petal-canvas";
    cv.setAttribute("aria-hidden", "true");
    document.body.appendChild(cv);
    var ctx = cv.getContext("2d"), W, H, dpr, list = [];
    var COLORS = ["#ffb7c5", "#ff9eb5", "#ffc8d6", "#ffd1dc", "#f7a1c4", "#ffe4ec"];
    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = window.innerWidth; H = window.innerHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function make(init) {
      return {
        x: Math.random() * W,
        y: init ? Math.random() * H : -20,
        s: 6 + Math.random() * 9,                 // 尺寸
        vy: 0.6 + Math.random() * 1.1,            // 下落速度
        vx: -0.4 + Math.random() * 1.2,           // 风
        sway: Math.random() * Math.PI * 2,
        swaySpeed: 0.01 + Math.random() * 0.02,
        rot: Math.random() * Math.PI * 2,
        vr: -0.03 + Math.random() * 0.06,
        flip: Math.random() * Math.PI * 2,        // 翻转（模拟 3D）
        vf: 0.02 + Math.random() * 0.04,
        c: COLORS[(Math.random() * COLORS.length) | 0],
        a: 0.65 + Math.random() * 0.35
      };
    }
    function drawPetal(p) {
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.scale(1, Math.abs(Math.cos(p.flip)) * 0.7 + 0.3);
      ctx.globalAlpha = p.a;
      ctx.fillStyle = p.c;
      var s = p.s;
      ctx.beginPath();                            // 樱花瓣：两条贝塞尔 + 顶部小缺口
      ctx.moveTo(0, s);
      ctx.bezierCurveTo(-s * 0.9, s * 0.4, -s * 0.7, -s * 0.8, -s * 0.15, -s);
      ctx.lineTo(0, -s * 0.75);
      ctx.lineTo(s * 0.15, -s);
      ctx.bezierCurveTo(s * 0.7, -s * 0.8, s * 0.9, s * 0.4, 0, s);
      ctx.fill();
      ctx.restore();
    }
    var count = function () { return Math.round(Math.min(60, Math.max(18, W / 28))); };
    function tick() {
      ctx.clearRect(0, 0, W, H);
      var n = count();
      while (list.length < n) list.push(make(list.length < n / 2));
      if (list.length > n) list.length = n;
      for (var i = 0; i < list.length; i++) {
        var p = list[i];
        p.sway += p.swaySpeed;
        p.x += p.vx + Math.sin(p.sway) * 0.8;
        p.y += p.vy;
        p.rot += p.vr;
        p.flip += p.vf;
        if (p.y > H + 20 || p.x < -30 || p.x > W + 30) list[i] = make(false);
        drawPetal(p);
      }
      if (!document.hidden) requestAnimationFrame(tick);
    }
    document.addEventListener("visibilitychange", function () { if (!document.hidden) requestAnimationFrame(tick); });
    window.addEventListener("resize", resize);
    resize();
    requestAnimationFrame(tick);
  })();

  /* ---------- 2. 卡片：滚动入场 + 3D 倾斜 + 光泽 ---------- */
  (function cards() {
    var cards = document.querySelectorAll(".category-tile, .article-link--card");
    if (!cards.length) return;
    var io = "IntersectionObserver" in window ? new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in-view"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 }) : null;
    cards.forEach(function (card, i) {
      card.classList.add("fx-card");
      card.style.setProperty("--delay", (i % 6) * 80 + "ms");
      if (io) io.observe(card); else card.classList.add("in-view");
      if (!finePointer) return;
      var glare = document.createElement("span");
      glare.className = "fx-glare";
      card.appendChild(glare);
      card.addEventListener("mousemove", function (e) {
        var r = card.getBoundingClientRect();
        var px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
        card.style.setProperty("--rx", ((0.5 - py) * 14).toFixed(2) + "deg");
        card.style.setProperty("--ry", ((px - 0.5) * 14).toFixed(2) + "deg");
        card.style.setProperty("--gx", (px * 100).toFixed(1) + "%");
        card.style.setProperty("--gy", (py * 100).toFixed(1) + "%");
        card.classList.add("tilting");
      });
      card.addEventListener("mouseleave", function () {
        card.classList.remove("tilting");
        card.style.setProperty("--rx", "0deg");
        card.style.setProperty("--ry", "0deg");
      });
    });
  })();

  /* ---------- 3. 打字机签名（打完停留，删掉，再打，循环） ---------- */
  (function typewriter() {
    var el = document.getElementById("typewriter");
    if (!el) return;
    var text = Array.from(el.getAttribute("data-text") || el.textContent.trim());
    var i = 0, deleting = false;
    el.textContent = "";
    function step() {
      if (!deleting) {
        i++;
        el.textContent = text.slice(0, i).join("");
        if (i >= text.length) { deleting = true; return setTimeout(step, 3500); }
        setTimeout(step, 140 + Math.random() * 120);
      } else {
        i--;
        el.textContent = text.slice(0, i).join("");
        if (i <= 0) { deleting = false; return setTimeout(step, 600); }
        setTimeout(step, 50);
      }
    }
    setTimeout(step, 500);
  })();

  /* ---------- 4. 鼠标：点击迸发 + 拖尾星光 ---------- */
  (function mouse() {
    var SYMBOLS = ["❤", "★", "✦", "✿", "♪", "❀"];
    var COLORS = ["#ff6b9d", "#ff9ec7", "#c084fc", "#7dd3fc", "#fcd34d", "#f472b6", "#a78bfa"];
    var pick = function (a) { return a[(Math.random() * a.length) | 0]; };
    function spawn(x, y, cls, txt, color, dx, dy, dur, size) {
      var s = document.createElement("span");
      s.className = cls;
      s.textContent = txt;
      s.style.left = x + "px";
      s.style.top = y + "px";
      s.style.color = color;
      s.style.fontSize = size + "px";
      s.style.setProperty("--dx", dx + "px");
      s.style.setProperty("--dy", dy + "px");
      s.style.animationDuration = dur + "ms";
      document.body.appendChild(s);
      setTimeout(function () { s.remove(); }, dur);
    }
    document.addEventListener("click", function (e) {
      var n = 8;
      for (var k = 0; k < n; k++) {
        var ang = (Math.PI * 2 * k) / n + Math.random() * 0.5;
        var dist = 40 + Math.random() * 40;
        spawn(e.clientX, e.clientY, "fx-burst", pick(SYMBOLS), pick(COLORS),
          Math.cos(ang) * dist, Math.sin(ang) * dist - 20, 800 + Math.random() * 300, 12 + Math.random() * 10);
      }
      var words = ["+1", "好耶", "喵~", "✧*｡", "(≧▽≦)", "冲鸭"];
      spawn(e.clientX, e.clientY, "fx-word", pick(words), pick(COLORS), 0, -60, 1000, 15);
    });
    if (!finePointer) return;
    var last = 0;
    document.addEventListener("mousemove", function (e) {
      var now = performance.now();
      if (now - last < 35) return;
      last = now;
      spawn(e.clientX, e.clientY, "fx-trail", pick(["✦", "·", "✧", "•"]), pick(COLORS),
        (Math.random() - 0.5) * 20, 15 + Math.random() * 20, 700, 8 + Math.random() * 8);
    }, { passive: true });
  })();
})();
