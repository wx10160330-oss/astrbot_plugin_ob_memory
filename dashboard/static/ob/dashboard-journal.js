/* 记忆桶页的手账排版：搜索框从全局头部挪进记忆桶页；状态/标签筛选和排序收进「筛选与排序」面板。
   搬动原有节点、复用原有筛选和操作处理函数；列表内编辑共享侧栏表单与保存校验，不改变后端接口。 */
(function () {
  'use strict';
  // 主页整体用 zoom 缩小（见 dashboard-journal.css）。实测 100vh/100vw 渲染后的实际尺寸，
  // 浏览器没按缩放换算时补一个系数，整屏高的侧栏不会在底部留空。
  (function measureViewportUnits() {
    var probe = document.createElement('div');
    probe.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:100vh;visibility:hidden;pointer-events:none';
    function measure() {
      if (!document.body) return;
      document.body.appendChild(probe);
      var rect = probe.getBoundingClientRect();
      probe.remove();
      if (!(rect.height > 0)) return;
      // vh 与 vw 随缩放的换算方式相同，用高度测得的系数同时补偿两者
      var factor = (window.innerHeight / rect.height).toFixed(4);
      document.documentElement.style.setProperty('--ob-vk', factor);
      document.documentElement.style.setProperty('--ob-wk', factor);
    }
    measure();
    window.addEventListener('resize', measure);
  })();
  var SORTS = [
    {value:'score', label:'综合分优先'},
    {value:'created_desc', label:'最新创建优先'},
    {value:'created_asc', label:'最早创建优先'}
  ];

  function mount() {
    var view = document.getElementById('list-view');
    var bulkbar = view && view.querySelector('.bucket-bulkbar');
    var filters = document.getElementById('filters');
    var sortSelect = document.getElementById('bucket-sort');
    var search = document.querySelector('.header .search-bar');
    if (!view || !bulkbar || !filters || !sortSelect || view.querySelector('.ob-list-top')) return;

    // 第一排：全选当前页 · 已选 N · 搜索
    var top = document.createElement('div');
    top.className = 'ob-list-top';
    var selectAll = document.getElementById('bucket-select-all');
    var selectLabel = selectAll && selectAll.closest('label');
    var count = document.getElementById('bucket-selected-count');
    if (selectLabel) top.appendChild(selectLabel);
    if (count) top.appendChild(count);
    if (search) { search.classList.add('ob-list-search'); top.appendChild(search); }
    view.insertBefore(top, bulkbar);

    // 第二排：筛选与排序 · 批量操作
    var wrap = document.createElement('div');
    wrap.className = 'ob-filter';
    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'ob-filter-trigger';
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', 'ob-filter-panel');
    trigger.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M7 12h10M10 18h4"/></svg>'
      + '<span>筛选与排序</span><em class="ob-filter-summary"></em>';
    var panel = document.createElement('div');
    panel.className = 'ob-filter-panel';
    panel.id = 'ob-filter-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', '筛选与排序');
    panel.hidden = true;
    panel.innerHTML = '<p class="ob-filter-title">排序 <i>Sort</i></p><div class="ob-filter-sorts"></div>'
      + '<p class="ob-filter-note" id="ob-filter-sort-note"></p>'
      + '<p class="ob-filter-title">记忆状态 <i>Status</i></p><div class="ob-filter-slot"></div>';
    wrap.appendChild(trigger);
    wrap.appendChild(panel);
    bulkbar.insertBefore(wrap, bulkbar.firstChild);
    // 批量按钮的文字包一层，窄屏时只留图标，保证和「筛选与排序」同一排
    bulkbar.querySelectorAll('.bucket-bulkbar-actions button').forEach(function (button) {
      var text = '';
      Array.from(button.childNodes).forEach(function (node) {
        if (node.nodeType === 3 && node.textContent.trim()) {
          text += node.textContent.trim();
          var span = document.createElement('span');
          span.className = 'ob-btn-text';
          span.textContent = node.textContent.trim();
          button.replaceChild(span, node);
        }
      });
      if (text) { button.setAttribute('aria-label', text); if (!button.title) button.title = text; }
    });

    // 原筛选容器整个搬进面板：buildFilters 仍按 id 重绘，委托点击仍有效；
    // 状态和标签按钮在同一个容器里，用 CSS order + 伪元素小标题分成两段。
    panel.querySelector('.ob-filter-slot').appendChild(filters);
    filters.classList.add('ob-filter-chips');

    var sorts = panel.querySelector('.ob-filter-sorts');
    SORTS.forEach(function (opt) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'ob-filter-sort';
      b.dataset.sort = opt.value;
      b.textContent = opt.label;
      b.addEventListener('click', function () {
        if (sortSelect.value === opt.value) return;
        if (typeof window.setBucketSort === 'function') window.setBucketSort(opt.value);
        else { sortSelect.value = opt.value; sortSelect.dispatchEvent(new Event('change')); }
        setTimeout(sync, 0);
      });
      sorts.appendChild(b);
    });

    function sync() {
      sorts.querySelectorAll('.ob-filter-sort').forEach(function (b) {
        var on = b.dataset.sort === sortSelect.value;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
      });
      var note = document.getElementById('bucket-sort-note');
      panel.querySelector('#ob-filter-sort-note').textContent = note ? note.textContent : '';
      var active = filters.querySelector('.filter-btn.active');
      var key = active ? active.dataset.filter : 'all';
      var parts = [];
      // 摘要只显示筛选条件；排序方式在面板里看，避免手机上按钮被撑长对不齐
      if (key && key !== 'all') parts.push(active.textContent.trim());
      trigger.querySelector('.ob-filter-summary').textContent = parts.join(' · ');
      trigger.classList.toggle('is-filtered', parts.length > 0);
    }
    // 筛选按钮由原脚本重绘；观察变化即可同步摘要
    new MutationObserver(sync).observe(filters, {childList:true, subtree:true, attributes:true, attributeFilter:['class']});
    var note = document.getElementById('bucket-sort-note');
    if (note) new MutationObserver(sync).observe(note, {childList:true, characterData:true, subtree:true});
    sync();

    function setOpen(open) {
      panel.hidden = !open;
      trigger.setAttribute('aria-expanded', String(open));
      wrap.classList.toggle('is-open', open);
      if (open) sync();
    }
    trigger.addEventListener('click', function () { setOpen(panel.hidden); });
    document.addEventListener('click', function (e) {
      if (!panel.hidden && !wrap.contains(e.target)) setOpen(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !panel.hidden) { setOpen(false); trigger.focus(); }
    });

    // 搜索框只在记忆桶页：在别的页按「/」时先切回记忆桶（捕获阶段先于原快捷键处理，
    // 否则隐藏页里的输入框拿不到焦点）
    document.addEventListener('keydown', function (e) {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || view.style.display !== 'none') return;
      var el = document.activeElement;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      var tab = document.querySelector('.tab[data-tab="list"]');
      if (tab) tab.click();
    }, true);

    // 便签编号接着页码往下数（第 2 页从 №11 开始），而不是每页都从 1 起
    var list = document.getElementById('bucket-list');
    if (list) {
      var number = function () {
        var page = typeof bucketPage === 'number' ? bucketPage : 1;
        var per = typeof BUCKETS_PER_PAGE === 'number' ? BUCKETS_PER_PAGE : 10;
        list.style.counterReset = 'ob-memo ' + (page - 1) * per;
      };
      new MutationObserver(number).observe(list, {childList:true});
      number();
    }
  }

  // ── 明暗切换：新主题从点击处像墨水一样晕开（View Transitions），不支持时整体颜色渐变 ──
  function smoothTheme() {
    var original = window.setDashboardColorMode;
    if (typeof original !== 'function' || original.obSmooth) return;
    var reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    var origin = null;
    document.addEventListener('pointerdown', function (e) { origin = {x:e.clientX, y:e.clientY, t:Date.now()}; }, true);
    var wrapped = function (value) {
      var before = document.documentElement.dataset.theme;
      var run = function () { original(value); };
      if (reduced.matches) { run(); return; }
      if (typeof document.startViewTransition !== 'function') {
        document.documentElement.classList.add('ob-theme-fading');
        run();
        setTimeout(function () { document.documentElement.classList.remove('ob-theme-fading'); }, 520);
        return;
      }
      var recent = origin && Date.now() - origin.t < 1500;
      var x = recent ? origin.x : window.innerWidth - 60, y = recent ? origin.y : 40;
      var transition = document.startViewTransition(run);
      transition.ready.then(function () {
        if (document.documentElement.dataset.theme === before) return;
        var r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
        document.documentElement.animate(
          {clipPath:['circle(0px at ' + x + 'px ' + y + 'px)', 'circle(' + r + 'px at ' + x + 'px ' + y + 'px)']},
          {duration:620, easing:'cubic-bezier(.45,.05,.25,1)', pseudoElement:'::view-transition-new(root)'});
      }).catch(function () {});
    };
    wrapped.obSmooth = true;
    window.setDashboardColorMode = wrapped;
  }

  // ── 每页的手账抬头：描边大编号 + 深色索引签 + 中文标题 + 法语小字 + 当日日期 ──
  var HEROES = {
    list:     {n:'01', kicker:'RECUEIL', zh:'记忆桶', fr:'Souvenirs gardés', en:'Memory buckets · 那些值得被记住的事'},
    network:  {n:'02', kicker:'GALAXIE', zh:'记忆星河', fr:'La voie lactée', en:'Memory galaxy · 想起哪颗，就点亮哪颗'},
    plan:     {n:'03', kicker:'PROMESSES', zh:'计划', fr:'Ce que je porte', en:'Plans · 答应过的、正在做的'},
    letters:  {n:'04', kicker:'LETTRES', zh:'信', fr:'Lettres à toi', en:'Letters · 写给彼此的话'},
    anchors:  {n:'05', kicker:'ANCRES', zh:'锚点', fr:'Points d’ancrage', en:'Anchors · 不会漂走的坐标'},
    'import': {n:'06', kicker:'ARCHIVES', zh:'导入', fr:'Importer le passé', en:'Import · 把旧日子搬进来'},
    logs:     {n:'07', kicker:'JOURNAL', zh:'日志', fr:'Journal de bord', en:'Logs · 运行的痕迹'},
    settings: {n:'05', kicker:'RÉGLAGES', zh:'设置', fr:'Petits réglages', en:'Settings · 让这里更像家'}
  };
  var DOW_FR = ['DIM.', 'LUN.', 'MAR.', 'MER.', 'JEU.', 'VEN.', 'SAM.'];
  var DOW_EN = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'];
  function heroes() {
    var now = new Date();
    var date = now.getFullYear() + '/' + String(now.getMonth() + 1).padStart(2, '0') + '/' + String(now.getDate()).padStart(2, '0');
    Object.keys(HEROES).forEach(function (key) {
      var view = document.getElementById(key + '-view');
      if (!view || view.querySelector(':scope > .ob-hero')) return;
      var h = HEROES[key];
      var el = document.createElement('header');
      el.className = 'ob-hero';
      el.dataset.view = key;
      el.innerHTML = '<span class="ob-hero-num" aria-hidden="true"></span>'
        + '<div class="ob-hero-main"><span class="ob-hero-kicker"></span><h2></h2><p class="ob-hero-en"></p></div>'
        + '<p class="ob-hero-fr" aria-hidden="true"></p>'
        + '<div class="ob-hero-date"><span class="d"></span><span class="w"></span><small></small></div>';
      el.querySelector('.ob-hero-num').textContent = h.n;
      el.querySelector('.ob-hero-kicker').textContent = h.kicker + ' · ' + h.n;
      el.querySelector('h2').textContent = h.zh;
      el.querySelector('.ob-hero-en').textContent = h.en;
      el.querySelector('.ob-hero-fr').textContent = h.fr;
      el.querySelector('.ob-hero-date .d').textContent = date;
      el.querySelector('.ob-hero-date .w').textContent = DOW_EN[now.getDay()];
      el.querySelector('.ob-hero-date small').textContent = DOW_FR[now.getDay()] + ' · ' + h.kicker.toLowerCase();
      view.insertBefore(el, view.firstChild);
    });
    // 记忆桶抬头的小字显示记忆总数
    var stats = document.getElementById('stats');
    var listHero = document.querySelector('#list-view > .ob-hero .ob-hero-date small');
    if (stats && listHero) {
      var count = function () {
        var total = typeof allBuckets !== 'undefined' && Array.isArray(allBuckets) ? allBuckets.length : 0;
        listHero.textContent = 'INDEX · ' + total + ' SOUVENIRS';
      };
      new MutationObserver(count).observe(stats, {childList:true, characterData:true, subtree:true});
      count();
    }
  }

  // 列表操作与行内编辑共享旧的表单和接口；刷新列表时移动现有节点保留草稿。
  function inlineMemories() {
    var view = document.getElementById('list-view');
    if (!view || view.dataset.inlineMemories) return;
    view.dataset.inlineMemories = '1';
    var editors = new Map(), messages = new Map(), busy = new Set(), sequence = 0;
    window.hasObInlineDrafts = function () { return Array.from(editors.values()).some(function (state) { return state.open; }); };
    window.discardObInlineDrafts = function () { editors.forEach(function (state) { state.host.remove(); }); editors.clear(); };
    function memories() { return typeof allBuckets !== 'undefined' && Array.isArray(allBuckets) ? allBuckets : []; }
    function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
    function decorate() {
      var data = memories();
      var live = new Set(data.map(function (b) { return String(b.id); }));
      editors.forEach(function (state, id) { if (!live.has(id)) { state.host.remove(); editors.delete(id); } });
      view.querySelectorAll('#bucket-list .bucket-row[data-id],#forgotten-list .bucket-row[data-id]').forEach(function (row) {
        var id = row.dataset.id;
        var memory = data.find(function (b) { return String(b.id) === id; });
        if (!memory) return;
        var actions = row.querySelector(':scope > .ob-row-actions');
        if (memory.letter_locked) {
          if (actions) actions.remove();
          if (editors.has(id)) { editors.get(id).host.remove(); editors.delete(id); }
          return;
        }
        if (!actions) {
          actions = document.createElement('div'); actions.className = 'ob-row-actions';
          ['pin','resolve','forget','archive','delete','edit'].forEach(function (action) {
            var button = document.createElement('button'); button.type = 'button'; button.dataset.action = action;
            actions.appendChild(button);
          });
          actions.addEventListener('click', function (event) {
            event.stopPropagation();
            var button = event.target.closest('button[data-action]');
            if (button && !button.disabled) act(id, button.dataset.action);
          });
          row.appendChild(actions);
        }
        var labels = {forget:memory.dont_surface ? '取消隐藏' : '隐藏',resolve:memory.resolved ? '激活' : '沉底',pin:memory.pinned ? '解钉' : '钉选',archive:'归档',delete:'删除',edit:'编辑'};
        actions.querySelectorAll('button').forEach(function (button) {
          setText(button, labels[button.dataset.action]); button.disabled = busy.has(id);
          if (button.dataset.action === 'delete') button.title = '永久删除，需再次确认';
          if (button.dataset.action === 'edit') {
            var state = editors.get(id);
            button.setAttribute('aria-expanded', String(!!(state && state.open)));
            if (state) button.setAttribute('aria-controls', state.host.id);
          }
        });
        var message = row.querySelector(':scope > .ob-row-message');
        if (messages.has(id)) {
          if (!message) { message = document.createElement('span'); message.className = 'ob-row-message'; message.setAttribute('role','status'); row.appendChild(message); }
          setText(message, messages.get(id));
        }
        var state = editors.get(id);
        if (state && state.host.parentNode !== row) {
          row.appendChild(state.host);
          var focused = state.focused;
          if (focused && document.activeElement === document.body) requestAnimationFrame(function () {
            if (focused.isConnected && document.activeElement === document.body) focused.focus({preventScroll:true});
          });
        }
      });
    }
    async function edit(id) {
      var state = editors.get(id);
      if (state && state.error && !state.open) { state.host.remove(); editors.delete(id); state = null; }
      if (state) { state.open = !state.open; state.host.hidden = !state.open; decorate(); return; }
      var host = document.createElement('div'); host.className = 'ob-inline-editor'; host.id = 'ob-inline-editor-' + (++sequence);
      host.textContent = '正在读取这条记忆…';
      host.addEventListener('click', function (event) { event.stopPropagation(); });
      host.addEventListener('focusin', function (event) { state.focused = event.target; });
      host.addEventListener('focusout', function () { if (host.isConnected) state.focused = null; });
      state = {host:host,open:true,focused:null,error:false}; editors.set(id,state); decorate();
      try {
        var response = await authFetch('/api/bucket/' + encodeURIComponent(id));
        if (!response) throw new Error('登录状态已失效，请重新登录。');
        var memory = await readJsonSafe(response);
        if (!response.ok) throw new Error(memory.error || ('HTTP ' + response.status));
        if (memory.letter_locked) throw new Error('这封信尚未开放，不能编辑。');
        if (editors.get(id) !== state) return;
        host.innerHTML = renderEditForm(id,Object.assign({},memory.metadata || {},{_content_for_edit:memory.content}),{idPrefix:host.id + '-'});
        ['edit-title','edit-why','edit-dont-surface','edit-weight'].forEach(function (name) {
          var input = host.querySelector('[data-edit-field="' + name + '"]');
          if (input) {
            var cell = input.parentElement.tagName === 'SPAN' ? input.parentElement : input;
            if (cell.previousElementSibling?.tagName === 'LABEL') cell.previousElementSibling.remove();
            cell.remove();
          }
        });
        host.querySelectorAll('[data-edit-field="edit-type"] option').forEach(function (option) {
          if (!['dynamic','permanent','feel','archived'].includes(option.value)) option.remove();
        });
        var form = host.querySelector('details'); form.open = true; form.dataset.inlineEditor = 'true';
        var bar = form.lastElementChild; bar.classList.add('ob-inline-savebar');
        var cancel = document.createElement('button'); cancel.type = 'button'; cancel.className = 'ob-inline-cancel'; cancel.textContent = '收起';
        cancel.addEventListener('click', function () { state.open = false; host.hidden = true; decorate(); });
        bar.insertBefore(cancel,bar.lastElementChild);
        form.addEventListener('ob-memory-saved', function () {
          editors.delete(id); messages.set(id,'已保存'); host.remove(); decorate();
        });
        if (state.open) form.querySelector('[data-edit-field=edit-name]').focus({preventScroll:true});
      } catch (error) {
        if (editors.get(id) !== state) return;
        state.error = true; host.textContent = '读取失败：' + error.message + '。请收起后重试。';
      }
    }
    async function act(id, action) {
      if (action === 'edit') return edit(id);
      if (action === 'delete') { openPermanentDelete([id]); return; }
      if (busy.has(id)) return;
      busy.add(id); decorate();
      try {
        var handler = {forget:window.bucketForget,resolve:window.bucketResolve,pin:window.bucketPin,archive:window.bucketArchive}[action];
        if (handler) await handler(id,{inline:true});
      } finally { busy.delete(id); decorate(); }
    }
    var observer = new MutationObserver(decorate);
    ['bucket-list','forgotten-list'].forEach(function (id) {
      var list = document.getElementById(id); if (list) observer.observe(list,{childList:true,subtree:true});
    });
    var cover = document.getElementById('auth-overlay');
    if (cover) new MutationObserver(function () {
      if (cover.style.display !== 'none') {
        editors.forEach(function (state) { state.host.remove(); }); editors.clear(); messages.clear(); busy.clear();
      }
    }).observe(cover,{attributes:true,attributeFilter:['style']});
    decorate();
  }

  function boot() { mount(); smoothTheme(); heroes(); inlineMemories(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, {once:true});
  else boot();
})();
