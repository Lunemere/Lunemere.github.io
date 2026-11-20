/* 霍夫曼树构建过程可视化（动态自底向上生长版）
 * 用法：页面里放 <div class="huffman"></div>，本脚本会把交互组件渲染进去。
 * 修改 DATA 即可换一组符号/频率。
 */
(function () {
  "use strict";

  /* ---- 数据：符号 + 频率（百分比） ---- */
  var DATA = [
    ["A", 7.7], ["B", 1.7], ["C", 3.2], ["D", 4.2], ["E", 12], ["F", 2.4],
    ["G", 1.7], ["H", 5], ["I", 7.6], ["J", 0.4], ["K", 0.7], ["L", 4.2],
    ["M", 2.4], ["N", 6.7], ["O", 6.7], ["P", 2], ["Q", 0.5], ["R", 5.9],
    ["S", 6.7], ["T", 8.5], ["U", 3.7], ["V", 1.2], ["W", 2.2], ["X", 0.4],
    ["Y", 2.2], ["Z", 0.2]
  ];

  var SVGNS = "http://www.w3.org/2000/svg";
  var GAP = 40;         // 同棵子树内相邻叶子间距
  var TREE_EXTRA = 20;  // 不同独立子树之间的额外间距
  var LEVEL = 58;       // 每层高度
  var PAD_X = 34;       // 水平画布留白
  var PAD_TOP = 30;     // 画布顶部留白
  var PAD_BOTTOM = 42;  // 画布底部留白（为叶子下方的概率文字留出空间）
  var R = 14;           // 结点半径
  var MIN_VB_W = 460;   // 初始缩放视口宽度（让前期少量结点更清晰）
  var ANIM_MS = 480;    // 每步位移动画时长（ms）
  var AUTO_MS = 820;    // 自动播放间隔（ms）

  function round1(v) { return Math.round(v * 10) / 10; }
  function fmt(v) { return String(round1(v)); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
  function easeOutBack(t) {
    var u = t - 1;
    return 1 + 2.1 * u * u * u + 1.1 * u * u;
  }
  function subT(raw, delay) {
    if (delay <= 0) return raw;
    if (delay >= 1) return 1;
    return Math.max(0, Math.min(1, (raw - delay) / (1 - delay)));
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS(SVGNS, name);
    if (attrs) for (var k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function cmp(a, b) { return a.w - b.w || a.id - b.id; }

  /* ---- 建树 ---- */
  function buildHuffman(data) {
    var id = 0;
    var leaves = data.map(function (d) {
      return { id: id++, ch: d[0], w: d[1], left: null, right: null };
    });
    var pool = leaves.slice();
    var steps = [];
    while (pool.length > 1) {
      pool.sort(cmp);
      var a = pool.shift();
      var b = pool.shift();
      var p = { id: id++, ch: null, w: round1(a.w + b.w), left: a, right: b };
      steps.push({ a: a, b: b, p: p });
      pool.push(p);
    }
    return { root: pool[0], leaves: leaves, steps: steps };
  }

  /* ---- 为最终树标注中序顺序与子树高度（保证森林左右有序、合并永不交叉） ---- */
  function annotate(root) {
    var order = 0;
    (function walk(n) {
      if (!n.left) {
        n.minOrder = order++;
        n.height = 0;
        return;
      }
      walk(n.left);
      walk(n.right);
      n.minOrder = n.left.minOrder;
      n.height = 1 + Math.max(n.left.height, n.right.height);
    })(root);
  }

  // 带权重的描述：A（7.7） / Σ41.4
  function nameOf(n) {
    return n.ch ? n.ch + "（" + fmt(n.w) + "）" : "Σ" + fmt(n.w);
  }

  function init(host) {
    if (host.getAttribute("data-ready")) return;
    host.setAttribute("data-ready", "1");

    var model = buildHuffman(DATA);
    var root = model.root;
    var steps = model.steps;
    var total = steps.length;
    var leafCount = model.leaves.length;
    annotate(root);

    var FULL_W = (leafCount - 1) * GAP + PAD_X * 2;
    var FULL_H = root.height * LEVEL + PAD_TOP + PAD_BOTTOM;
    var AR = FULL_W / FULL_H;
    var reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /* ---------- DOM 骨架 ---------- */
    var bar = document.createElement("div");
    bar.className = "h-bar";
    bar.innerHTML =
      '<button type="button" class="h-btn" data-act="reset">重置</button>' +
      '<button type="button" class="h-btn h-btn--primary" data-act="next">下一步 ›</button>' +
      '<button type="button" class="h-btn" data-act="prev">‹ 上一步</button>' +
      '<button type="button" class="h-btn" data-act="play">自动播放</button>' +
      '<span class="h-progress"></span>' +
      '<label class="h-switch"><input type="checkbox" data-act="flip"> 左 0 / 右 1</label>';

    var status = document.createElement("p");
    status.className = "h-status";

    var hint = document.createElement("p");
    hint.className = "h-hint";

    var stage = document.createElement("div");
    stage.className = "h-stage";

    var canvas = document.createElement("div");
    canvas.className = "h-canvas";

    var emptyTip = document.createElement("div");
    emptyTip.className = "h-empty";
    emptyTip.textContent = "点击「下一步」或「自动播放」，按右侧概率从小到大逐步合并，自底向上生长出霍夫曼树";

    var rail = document.createElement("aside");
    rail.className = "h-rail";
    rail.setAttribute("aria-label", "剩余待合并结点按概率升序列表");
    rail.innerHTML =
      '<div class="h-rail-inner">' +
        '<div class="h-rail-head">' +
          '<span class="h-rail-title"></span>' +
          '<span class="h-rail-order">小 ↓ 大</span>' +
        '</div>' +
        '<div class="h-rail-list"></div>' +
        '<div class="h-rail-legend">' +
          '<span><i class="h-rail-dot"></i>未入图</span>' +
          '<span><i class="h-rail-dot h-rail-dot--inner"></i>子树根</span>' +
        '</div>' +
      '</div>';

    var railTitle = rail.querySelector(".h-rail-title");
    var railList = rail.querySelector(".h-rail-list");

    var codes = document.createElement("div");
    codes.className = "h-codes";

    stage.appendChild(canvas);
    stage.appendChild(rail);

    host.appendChild(bar);
    host.appendChild(status);
    host.appendChild(hint);
    host.appendChild(stage);
    host.appendChild(codes);

    /* ---------- 画布 ---------- */
    var svg = svgEl("svg", {
      role: "img",
      "aria-label": "霍夫曼树的逐步构建过程"
    });
    svg.style.aspectRatio = FULL_W + " / " + FULL_H;

    var ground = svgEl("line", {
      y1: 33, y2: 33,
      class: "h-ground"
    });
    var gEdges = svgEl("g");
    var gNodes = svgEl("g");
    svg.appendChild(ground);
    svg.appendChild(gEdges);
    svg.appendChild(gNodes);
    canvas.appendChild(svg);
    canvas.appendChild(emptyTip);

    var leftBit = "1"; // 题目约定：左 1 右 0

    var nodeEls = {};  // id -> <g>
    var stepEls = [];  // [{ a: { line, label }, b: { line, label } }]

    // 预创建边与位标签（初始隐藏，位置随动画帧实时同步）
    steps.forEach(function () {
      var pack = {};
      ["a", "b"].forEach(function (key) {
        var line = svgEl("line", { class: "h-edge" });
        line.style.opacity = "0";
        line.style.visibility = "hidden";
        var t = svgEl("text", { class: "h-bit" });
        t.textContent = key === "a" ? leftBit : (leftBit === "1" ? "0" : "1");
        t.style.opacity = "0";
        t.style.visibility = "hidden";
        gEdges.appendChild(line);
        gEdges.appendChild(t);
        pack[key] = { line: line, label: t };
      });
      stepEls.push(pack);
    });

    var allNodes = model.leaves.concat(steps.map(function (s) { return s.p; }));
    allNodes.forEach(function (n) {
      var g = svgEl("g", {
        class: "h-node " + (n.ch ? "h-node--leaf" : "h-node--inner")
      });
      g.style.opacity = "0";
      g.style.visibility = "hidden";
      g.appendChild(svgEl("circle", { class: "h-halo", r: 18 }));
      g.appendChild(svgEl("circle", { class: "h-dot", r: R }));
      var t = svgEl("text", { class: "h-label", y: 4 });
      t.textContent = n.ch ? n.ch : fmt(n.w);
      if (!n.ch) t.classList.add("h-label--w");
      g.appendChild(t);
      if (n.ch) {
        var prob = svgEl("text", { class: "h-prob", y: 25 });
        prob.textContent = fmt(n.w) + "%";
        g.appendChild(prob);
      }
      gNodes.appendChild(g);
      nodeEls[n.id] = g;
    });

    /* ---------- 动态布局计算 ---------- */
    function forestAt(k) {
      var pool = model.leaves.slice();
      for (var i = 0; i < k; i++) {
        var s = steps[i];
        pool = pool.filter(function (n) { return n.id !== s.a.id && n.id !== s.b.id; });
        pool.push(s.p);
      }
      pool.sort(cmp);
      return pool;
    }

    // 仅布局已经参与合并的树：叶子随合并逐步出现，子树底对齐并向上生长
    function computeLayout(k) {
      var pool = forestAt(k);
      var built = pool
        .filter(function (n) { return !!n.left; })
        .sort(function (a, b) { return a.minOrder - b.minOrder; });

      var pos = {};
      var cursor = 0;
      var maxH = 0;

      built.forEach(function (tree, idx) {
        if (idx > 0) cursor += TREE_EXTRA;
        if (tree.height > maxH) maxH = tree.height;
        (function place(n, depth) {
          var y = -(tree.height - depth) * LEVEL;
          if (n.left) {
            place(n.left, depth + 1);
            place(n.right, depth + 1);
            pos[n.id] = {
              x: (pos[n.left.id].x + pos[n.right.id].x) / 2,
              y: y
            };
          } else {
            pos[n.id] = { x: cursor, y: y };
            cursor += GAP;
          }
        })(tree, 0);
      });

      var vbW = MIN_VB_W;
      if (built.length > 0) {
        var spanW = Math.max(0, cursor - GAP);
        var half = spanW / 2;
        for (var id in pos) pos[id].x -= half;

        var spanH = maxH * LEVEL;
        vbW = Math.max(MIN_VB_W, spanW + PAD_X * 2);
        var vbH = vbW / AR;
        if (spanH + PAD_TOP + PAD_BOTTOM > vbH) {
          vbH = spanH + PAD_TOP + PAD_BOTTOM;
          vbW = vbH * AR;
        }
      }

      return { pool: pool, built: built, pos: pos, vbW: vbW };
    }

    /* ---------- 逐帧补间动画引擎 ---------- */
    var nodeState = {};
    allNodes.forEach(function (n) {
      nodeState[n.id] = {
        x: 0, y: 0, s: 0, o: 0,
        sx: 0, sy: 0, ss: 0, so: 0,
        tx: 0, ty: 0, ts: 0, to: 0,
        delay: 0, entering: false, exiting: false
      };
    });

    var edgeState = steps.map(function () {
      return { curr: 0, start: 0, target: 0, delay: 0 };
    });

    var cam = { curr: MIN_VB_W, start: MIN_VB_W, target: MIN_VB_W };
    var groundAlpha = { curr: 0, start: 0, target: 0 };
    var rafId = null;

    function applyFrame(rawT) {
      var eCam = easeOutCubic(rawT);
      cam.curr = lerp(cam.start, cam.target, eCam);
      var vbW = cam.curr;
      var vbH = vbW / AR;
      var vbX = -vbW / 2;
      var vbY = PAD_BOTTOM - vbH;
      svg.setAttribute(
        "viewBox",
        vbX.toFixed(2) + " " + vbY.toFixed(2) + " " + vbW.toFixed(2) + " " + vbH.toFixed(2)
      );

      groundAlpha.curr = lerp(groundAlpha.start, groundAlpha.target, eCam);
      ground.setAttribute("x1", (vbX + 16).toFixed(1));
      ground.setAttribute("x2", (vbX + vbW - 16).toFixed(1));
      ground.style.opacity = (groundAlpha.curr * 0.45).toFixed(3);

      // 1. 先更新结点位置、缩放与透明度
      for (var i = 0; i < allNodes.length; i++) {
        var n = allNodes[i];
        var st = nodeState[n.id];
        if (st.so === 0 && st.to === 0) continue;

        var lt = subT(rawT, st.delay);
        var ePos = easeOutCubic(lt);
        var eScale = st.entering ? easeOutBack(lt) : ePos;

        st.x = lerp(st.sx, st.tx, ePos);
        st.y = lerp(st.sy, st.ty, ePos);
        st.s = lerp(st.ss, st.ts, eScale);
        st.o = lerp(st.so, st.to, ePos);

        var g = nodeEls[n.id];
        if (st.o <= 0.005) {
          g.style.opacity = "0";
          g.style.visibility = "hidden";
        } else {
          g.style.visibility = "visible";
          g.style.opacity = st.o.toFixed(3);
          g.setAttribute(
            "transform",
            "translate(" + st.x.toFixed(2) + "," + st.y.toFixed(2) +
            ") scale(" + Math.max(0.01, st.s).toFixed(3) + ")"
          );
        }
      }

      // 2. 再按结点实时坐标同步所有连边与 0/1 标签（永不脱节）
      for (var j = 0; j < steps.length; j++) {
        var es = edgeState[j];
        if (es.start === 0 && es.target === 0) continue;

        var elt = subT(rawT, es.delay);
        es.curr = lerp(es.start, es.target, easeOutCubic(elt));

        var pack = stepEls[j];
        if (es.curr <= 0.005) {
          pack.a.line.style.visibility = "hidden";
          pack.b.line.style.visibility = "hidden";
          pack.a.label.style.visibility = "hidden";
          pack.b.label.style.visibility = "hidden";
          pack.a.line.style.opacity = "0";
          pack.b.line.style.opacity = "0";
          pack.a.label.style.opacity = "0";
          pack.b.label.style.opacity = "0";
          continue;
        }

        var s = steps[j];
        var ps = nodeState[s.p.id];
        var pairs = [[s.a, pack.a], [s.b, pack.b]];
        var edgeOp = (es.curr * 0.58).toFixed(3);
        var bitOp = Math.max(0, (es.curr - 0.25) / 0.75);

        for (var pIdx = 0; pIdx < 2; pIdx++) {
          var cs = nodeState[pairs[pIdx][0].id];
          var el = pairs[pIdx][1];

          el.line.style.visibility = "visible";
          el.line.setAttribute("x1", ps.x.toFixed(2));
          el.line.setAttribute("y1", ps.y.toFixed(2));
          el.line.setAttribute("x2", cs.x.toFixed(2));
          el.line.setAttribute("y2", cs.y.toFixed(2));
          el.line.style.opacity = edgeOp;

          if (bitOp > 0.01) {
            var mx = (ps.x + cs.x) / 2 + (cs.x >= ps.x ? 9 : -9);
            var my = (ps.y + cs.y) / 2 + 4;
            el.label.style.visibility = "visible";
            el.label.setAttribute("x", mx.toFixed(2));
            el.label.setAttribute("y", my.toFixed(2));
            el.label.style.opacity = bitOp.toFixed(3);
          } else {
            el.label.style.visibility = "hidden";
            el.label.style.opacity = "0";
          }
        }
      }
    }

    function animateTo(layout, prevStep, nextStep, instant) {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }

      var isSingleForward = nextStep === prevStep + 1;
      var merged = isSingleForward ? steps[nextStep - 1] : null;

      cam.start = instant ? layout.vbW : cam.curr;
      cam.target = layout.vbW;

      var wantGround = nextStep > 0 ? 1 : 0;
      groundAlpha.start = instant ? wantGround : groundAlpha.curr;
      groundAlpha.target = wantGround;

      for (var i = 0; i < allNodes.length; i++) {
        var n = allNodes[i];
        var st = nodeState[n.id];
        var tpos = layout.pos[n.id];

        if (tpos) {
          if (!instant && st.o > 0.05 && !st.exiting) {
            // 已在画布上的结点：从当前动画位置平滑滑向新位置
            st.sx = st.x; st.sy = st.y; st.ss = st.s; st.so = st.o;
            st.tx = tpos.x; st.ty = tpos.y; st.ts = 1; st.to = 1;
            st.delay = 0; st.entering = false; st.exiting = false;
          } else {
            // 新加入画布的结点：从下方或两子之间升起
            st.tx = tpos.x; st.ty = tpos.y; st.ts = 1; st.to = 1;
            st.entering = true; st.exiting = false;
            if (merged && n.id === merged.p.id) {
              var ca = layout.pos[merged.a.id];
              var cb = layout.pos[merged.b.id];
              st.sx = (ca.x + cb.x) / 2;
              st.sy = (ca.y + cb.y) / 2 - 10;
              st.ss = 0.2; st.so = 0; st.delay = 0.14;
            } else if (!n.left) {
              // 叶子首次从下方队列升入画布
              st.sx = tpos.x;
              st.sy = tpos.y + 34;
              st.ss = 0.25; st.so = 0; st.delay = 0;
            } else {
              st.sx = tpos.x;
              st.sy = tpos.y + 20;
              st.ss = 0.3; st.so = 0; st.delay = 0.08;
            }
            st.x = st.sx; st.y = st.sy; st.s = st.ss; st.o = st.so;
          }
        } else {
          if (!instant && st.o > 0.01) {
            // 回退或重置时平滑淡出下沉
            st.sx = st.x; st.sy = st.y; st.ss = st.s; st.so = st.o;
            st.tx = st.x; st.ty = st.y + 20; st.ts = 0.25; st.to = 0;
            st.delay = 0; st.entering = false; st.exiting = true;
          } else {
            st.x = 0; st.y = 0; st.s = 0; st.o = 0;
            st.sx = 0; st.sy = 0; st.ss = 0; st.so = 0;
            st.tx = 0; st.ty = 0; st.ts = 0; st.to = 0;
            st.delay = 0; st.entering = false; st.exiting = false;
            var g = nodeEls[n.id];
            g.style.opacity = "0";
            g.style.visibility = "hidden";
          }
        }
      }

      for (var j = 0; j < steps.length; j++) {
        var es = edgeState[j];
        var wantOn = j < nextStep;
        es.start = instant ? (wantOn ? 1 : 0) : es.curr;
        es.target = wantOn ? 1 : 0;
        es.delay = (!instant && wantOn && es.curr < 0.05 && isSingleForward && j === nextStep - 1)
          ? 0.14 : 0;
      }

      if (instant || reduceMotion) {
        applyFrame(1);
        return;
      }

      var startTime = performance.now();
      function tick(now) {
        var rawT = Math.min(1, (now - startTime) / ANIM_MS);
        applyFrame(rawT);
        if (rawT < 1) {
          rafId = requestAnimationFrame(tick);
        } else {
          rafId = null;
        }
      }
      applyFrame(0);
      rafId = requestAnimationFrame(tick);
    }

    /* ---------- 状态渲染 ---------- */
    var current = 0;
    var timer = null;

    function render(prevStep, instant) {
      var layout = computeLayout(current);
      animateTo(layout, prevStep, current, !!instant);

      emptyTip.classList.toggle("is-hidden", current > 0);

      var pick = current < total ? steps[current] : null;
      var latest = current > 0 ? steps[current - 1].p : null;

      // 高亮：刚合成的父结点（is-latest）+ 下一步将合并的已在画布上的结点（is-pick）
      allNodes.forEach(function (n) {
        var g = nodeEls[n.id];
        var onCanvas = !!layout.pos[n.id];
        var hot = onCanvas && pick && (n.id === pick.a.id || n.id === pick.b.id);
        var isLatest = onCanvas && latest && n.id === latest.id;
        g.classList.toggle("is-latest", !!isLatest);
        g.classList.toggle("is-pick", !!hot);
      });

      // 文字
      if (current === 0) {
        status.textContent = "初始状态：画布为空，共 " + leafCount + " 个符号待按概率从小到大合并。";
      } else {
        var last = steps[current - 1];
        status.textContent = "第 " + current + " 次合并：" + nameOf(last.a) + " + " +
          nameOf(last.b) + " → Σ" + fmt(last.p.w) + "。";
      }

      if (!pick) {
        hint.textContent = "建树完成，共 " + total + " 次合并。下面可以读出每个符号的编码。";
      } else if (current === 0) {
        hint.textContent = "下一步：取出概率最小的 " + nameOf(pick.a) + " 与 " +
          nameOf(pick.b) + " 开始建树。";
      } else {
        hint.textContent = "下一步：合并 " + nameOf(pick.a) + " 与 " + nameOf(pick.b) + "。";
      }

      host.querySelector(".h-progress").textContent = current + " / " + total;

      // 右侧剩余概率点列表（从上到下概率递增）
      railTitle.textContent = pick ? "待合并 (" + layout.pool.length + ")" : "根节点 (1)";
      railList.innerHTML = "";
      var maxW = layout.pool.length ? layout.pool[layout.pool.length - 1].w : 1;
      layout.pool.forEach(function (n) {
        var isInner = !!n.left;
        var isPick = pick && (n.id === pick.a.id || n.id === pick.b.id);
        var isNew = latest && n.id === latest.id;
        var item = document.createElement("div");
        item.className = "h-rail-item" +
          (isInner ? " h-rail-item--inner" : "") +
          (isNew ? " is-latest" : "") +
          (isPick ? " is-pick" : "");
        item.style.setProperty("--p", Math.max(4, (n.w / maxW) * 100).toFixed(1) + "%");
        item.innerHTML =
          '<span class="h-rail-dot' + (isInner ? " h-rail-dot--inner" : "") + '"></span>' +
          '<span class="h-rail-name">' + (n.ch ? n.ch : "Σ") + '</span>' +
          '<span class="h-rail-w">' + fmt(n.w) + '%</span>';
        railList.appendChild(item);
      });
      railList.scrollTop = 0;

      // 编码表
      if (!pick) {
        showCodes();
        codes.classList.add("is-on");
      } else {
        codes.classList.remove("is-on");
      }

      bar.querySelector('[data-act="prev"]').disabled = current === 0;
      bar.querySelector('[data-act="next"]').disabled = current === total;
    }

    function showCodes() {
      var codesMap = {};
      var l = leftBit, r = leftBit === "1" ? "0" : "1";
      (function walk(n, prefix) {
        if (!n.left) { codesMap[n.ch] = prefix; return; }
        walk(n.left, prefix + l);
        walk(n.right, prefix + r);
      })(root, "");

      var totalW = 0, cost = 0, maxLen = 0;
      DATA.forEach(function (d) {
        totalW += d[1];
        cost += d[1] * codesMap[d[0]].length;
        maxLen = Math.max(maxLen, codesMap[d[0]].length);
      });
      var avg = cost / totalW;
      var fixed = Math.ceil(Math.log(leafCount) / Math.LN2);

      codes.innerHTML =
        '<div class="h-codes-head">各符号编码（左 ' + l + ' / 右 ' + r + '，叶子向上读到根）</div>' +
        '<div class="h-code-grid">' +
        DATA.map(function (d) {
          return '<span class="h-code"><b>' + d[0] + '</b>' + fmt(d[1]) +
            ' <span class="h-arrow">→</span> <code>' + codesMap[d[0]] + '</code></span>';
        }).join("") +
        "</div>" +
        '<div class="h-codes-foot">平均码长 ' + avg.toFixed(3) + " 位/符号 · 定长编码需要 " + fixed +
        " 位 · 最长码 " + maxLen + " 位</div>";
    }

    /* ---------- 交互 ---------- */
    function go(k, instant) {
      var prev = current;
      current = Math.max(0, Math.min(total, k));
      render(prev, instant);
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      var btn = bar.querySelector('[data-act="play"]');
      btn.textContent = "自动播放";
    }

    bar.addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-act]");
      if (!btn) return;
      var act = btn.getAttribute("data-act");
      if (act === "next") { stop(); go(current + 1); }
      else if (act === "prev") { stop(); go(current - 1); }
      else if (act === "reset") { stop(); go(0); }
      else if (act === "play") {
        if (timer) { stop(); return; }
        if (current === total) go(0);
        btn.textContent = "暂停 ❚❚";
        timer = setInterval(function () {
          if (current >= total) { stop(); return; }
          go(current + 1);
        }, AUTO_MS);
      }
    });

    bar.querySelector('[data-act="flip"]').addEventListener("change", function () {
      leftBit = this.checked ? "0" : "1";
      var r = leftBit === "1" ? "0" : "1";
      steps.forEach(function (s, i) {
        stepEls[i].a.label.textContent = leftBit;
        stepEls[i].b.label.textContent = r;
      });
      if (current === total) showCodes();
    });

    go(0, true);
  }

  function boot() {
    var hosts = document.querySelectorAll(".huffman");
    for (var i = 0; i < hosts.length; i++) init(hosts[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
