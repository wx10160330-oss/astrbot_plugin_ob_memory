/* AstrBot 插件适配：保留 OB 外观及交互，接口与插件原界面隔离。 */
(function () {
  'use strict';
  var rawFetch = window.fetch.bind(window);
  var selectedSession = '';
  var sessionPromise = null;
  var sessionData = null;
  var dirty = false;
  var saving = false;
  var sessionRevision = 0;
  var searchRevision = 0;
  var nativeDetail;
  var unsupported = {
    about: 'OB 界面来自本地 Ombre Brain 3.11.0，记忆与模型调用继续由 AstrBot 插件提供。',
    'v3-debug': '插件未提供 OB 的调试回放。'
  };
  try { selectedSession = localStorage.getItem('memory_dashboard_session_id') || ''; } catch (_) {}

  function feedback(message) {
    var el = document.getElementById('ob-feedback');
    if (el) el.textContent = message;
  }
  function canLeaveEdit(message) {
    if (saving || (window.isObDomainSaving && window.isObDomainSaving())) { feedback('正在保存，请稍后操作。'); return false; }
    var domainDraft = window.hasObDomainDrafts && window.hasObDomainDrafts();
    var inlineDraft = window.hasObInlineDrafts && window.hasObInlineDrafts();
    if ((dirty || domainDraft || inlineDraft) && !confirm(message)) return false;
    if (domainDraft && window.discardObDomainDrafts) window.discardObDomainDrafts();
    if (inlineDraft && window.discardObInlineDrafts) window.discardObInlineDrafts();
    dirty = false;
    return true;
  }
  function rememberSession(value) {
    selectedSession = value || '';
    try { localStorage.setItem('memory_dashboard_session_id', selectedSession); } catch (_) {}
  }
  async function ensureSession() {
    if (!sessionPromise) {
      sessionPromise = rawFetch('/ob/api/sessions', {cache:'no-store'}).then(async function (response) {
        if (!response.ok) { sessionPromise = null; return null; }
        sessionData = await response.json();
        var rows = sessionData.sessions;
        var valid = rows.some(function (s) { return s.session_id === selectedSession; });
        rememberSession(!sessionData.locked && valid ? selectedSession : sessionData.current);
        renderSessions();
        return sessionData;
      }).catch(function (error) { sessionPromise = null; throw error; });
    }
    return sessionPromise;
  }
  window.fetch = async function (input, init) {
    var url = new URL(input instanceof Request ? input.url : String(input), location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith('/api/')) return rawFetch(input, init);
    // 等会话确认后再发业务请求；不会先短暂显示另一会话的数据。
    var sessions = await ensureSession();
    if (!sessions) return new Response(JSON.stringify({error:'请先登录',detail:'auth required'}),
      {status:401,headers:{'Content-Type':'application/json'}});
    url.pathname = '/ob' + url.pathname;
    if (selectedSession) url.searchParams.set('session', selectedSession);
    return rawFetch(input instanceof Request ? new Request(url, input) : url, init);
  };

  function renderSessions() {
    var picker = document.getElementById('ob-session');
    if (!picker || !sessionData) return;
    picker.replaceChildren();
    sessionData.sessions.forEach(function (session) {
      var option = document.createElement('option');
      option.value = session.session_id;
      option.textContent = session.session_id + ' · ' + session.memory_count + ' 条记忆';
      picker.appendChild(option);
    });
    if (!sessionData.sessions.length) {
      var empty = document.createElement('option');
      empty.textContent = '暂无会话，先和 AI 留下一段记忆';
      picker.appendChild(empty);
    }
    picker.value = selectedSession;
    picker.disabled = sessionData.locked || !sessionData.sessions.length;
    document.getElementById('ob-session-note').textContent = sessionData.locked ? '已锁定到插件配置的会话' : '与原界面共用会话和记忆';
  }

  function showView(target) {
    ['list','breath','network','plan','import','faq','logs','v3-debug','settings','letters','anchors','about'].forEach(function (id) {
      var el = document.getElementById(id + '-view');
      if (el) el.style.display = id === target && !unsupported[target] ? '' : 'none';
    });
    document.querySelectorAll('.tab').forEach(function (tab) { tab.classList.toggle('active', tab.dataset.tab === target); });
    var notice = document.getElementById('ob-unavailable');
    notice.hidden = !unsupported[target];
    if (unsupported[target]) document.getElementById('ob-unavailable-text').textContent = unsupported[target];
    if (target !== 'network' && typeof _netRAF !== 'undefined' && _netRAF) {
      cancelAnimationFrame(_netRAF); _netRAF = null;
    }
    if (window.setObGalaxyActive) window.setObGalaxyActive(target === 'network');
  }

  function disableField(id) {
    var input = document.getElementById(id);
    if (!input) return;
    var cell = input.parentElement.tagName === 'SPAN' ? input.parentElement : input;
    if (cell.previousElementSibling && cell.previousElementSibling.tagName === 'LABEL') cell.previousElementSibling.remove();
    cell.remove();
  }
  function customizeDetail() {
    ['edit-title','edit-why','edit-dont-surface'].forEach(disableField);
    var type = document.getElementById('edit-type');
    if (type) Array.from(type.options).forEach(function (option) {
      if (!['dynamic','permanent','feel','archived'].includes(option.value)) option.remove();
    });
    var form = document.getElementById('bucket-edit-form');
    if (form) form.addEventListener('input', function () { dirty = true; });
    var actions = document.querySelector('#detail-content .detail-actions');
    if (actions) {
      actions.querySelectorAll('[onclick*="bucketAnchor"],[onclick*="bucketForget"]').forEach(function (el) { el.remove(); });
    }
    document.querySelectorAll('#detail-content .detail-meta .field').forEach(function (field) {
      var label = field.querySelector('label');
      if (label && label.textContent === 'Anchor') field.remove();
    });
  }

  async function saveEdit(id, editorRoot) {
    if (saving) return;
    var inline = !!(editorRoot && editorRoot.dataset.inlineEditor === 'true');
    var field = function (name) {
      return inline ? editorRoot.querySelector('[data-edit-field="edit-' + name + '"]') : document.getElementById('edit-' + name);
    };
    var msg = inline ? editorRoot.querySelector('[data-edit-field="edit-msg"]') : document.getElementById('edit-msg');
    var form = inline ? editorRoot : document.getElementById('bucket-edit-form');
    var body = {};
    ['name','content','domain','tags'].forEach(function (key) { body[key] = field(key).value; });
    ['pinned','resolved','digested'].forEach(function (key) { body[key] = field(key).checked; });
    body.importance = Number(field('importance').value);
    var type = field('type');
    if (type && !type.disabled) body.type = type.value;
    if (!body.name.trim() || !body.content.trim()) { msg.textContent = '名称和正文不能为空。'; return; }
    saving = true;
    form.inert = true;
    msg.textContent = '保存中…';
    try {
      var response = await fetch('/api/bucket/' + encodeURIComponent(id) + '/edit',
        {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      var result = await response.json();
      if (!response.ok) throw new Error(result.error || '保存失败');
      dirty = false;
      if (inline) form.dispatchEvent(new CustomEvent('ob-memory-saved', {bubbles:true,detail:{id:id}}));
      else if (await nativeDetail(id)) customizeDetail();
      await loadBuckets();
      feedback('记忆已保存。' + (result.warnings || []).join(' '));
    } catch (error) { msg.textContent = error.message; }
    finally { saving = false; if (form.isConnected) form.inert = false; }
  }

  function install() {
    if (document.body.classList.contains('astrbot-ob')) return;
    document.body.classList.add('astrbot-ob');
    ['breath','import','logs','anchors'].forEach(function (id) {
      var tab = document.querySelector('.tab[data-tab="' + id + '"]');
      var view = document.getElementById(id + '-view');
      if (tab) tab.remove();
      if (view) view.remove();
    });
    document.querySelectorAll('[onclick*="hardDelete"], [onclick*="runBucketBatch(\'forget\')"]').forEach(function (button) { button.remove(); });
    window.refreshAstrbotSessions = async function () { sessionPromise = null; await ensureSession(); };
    var bar = document.createElement('div');
    bar.id = 'ob-session-bar';
    bar.innerHTML = '<label for="ob-session">当前会话</label><select id="ob-session" aria-label="当前会话"></select><span id="ob-session-note"></span>';
    var counters = [
      ['total','总记忆',function () { return true; }],
      ['pinned','核心准则',function (b) { return b.pinned; }],
      ['feel','感受',function (b) { return b.type === 'feel'; }],
      ['active','活跃中',function (b) { return !b.resolved && !b.pinned && !b.digested && b.type !== 'feel' && b.type !== 'archived'; }],
      ['dormant','已沉底',function (b) { return (b.type === 'archived' || b.resolved) && b.type !== 'feel'; }],
      ['digested','已隐藏',function (b) { return b.digested; }]
    ];
    var stats = document.createElement('section');
    stats.id = 'ob-memory-stats'; stats.setAttribute('aria-label','当前会话记忆统计');
    stats.innerHTML = '<dl>' + counters.map(function (item) {
      return '<div class="ob-stat" data-stat="' + item[0] + '"><dt>' + item[1] + '</dt><dd>—</dd></div>';
    }).join('') + '</dl>';
    document.querySelector('.tabs').after(stats);
    stats.after(bar);
    var originalStats = window.updateStats;
    window.updateStats = function () {
      originalStats();
      counters.forEach(function (item) {
        stats.querySelector('[data-stat="' + item[0] + '"] dd').textContent = allBuckets.filter(item[2]).length;
      });
    };
    var status = document.createElement('p');
    status.id = 'ob-feedback'; status.setAttribute('role','status');
    bar.after(status);
    var unavailable = document.createElement('section');
    unavailable.id = 'ob-unavailable'; unavailable.className = 'content'; unavailable.hidden = true;
    unavailable.innerHTML = '<div class="config-section"><h2>此功能暂未适配</h2><p id="ob-unavailable-text"></p><p>点击顶部的切换按钮，可继续使用原界面。</p></div>';
    status.after(unavailable);
    var summary = document.createElement('section');
    summary.id = 'ob-plugin-settings'; summary.className = 'config-section';
    summary.innerHTML = '<h3>AstrBot 插件</h3><p>模型、自动记录和会话隔离请在 AstrBot 的插件配置中管理。整理记录与导入导出可切换到原界面使用。</p><p id="ob-plugin-summary"></p>';
    document.getElementById('sec-appearance').after(summary);
    renderSessions();
    document.getElementById('ob-session').addEventListener('change', async function (event) {
      var picker = event.target;
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并切换会话？')) { picker.value = selectedSession; return; }
      sessionRevision += 1;
      searchRevision += 1;
      document.dispatchEvent(new Event('ob-session-changed'));
      rememberSession(picker.value);
      closeDetail(); selectedBucketIds.clear();
      bucketLoadGeneration += 1;
      // 清空上一会话的列表、选择与图；异步响应不能继续呈现旧会话。
      allBuckets = []; currentFilter = 'active';
      document.getElementById('search-input').value = '';
      renderBuckets([]); updateStats();
      showView('list');
      await loadBuckets();
      feedback('已切换会话。');
    });

    // 原版启动钩子在认证之后执行，这些功能由宿主负责，关闭独立 OB 的调用。
    ['syncRestartRequirement','refreshAnchorCounter','loadOwnerBadge','pollCriticalErrors','maybeShowOnboarding','initSelfFab'].forEach(function (key) {
      window[key] = async function () {};
    });
    window.activateDashboardTab = function (tab) {
      var target = tab.dataset.tab;
      showView(target);
      if (target === 'list') return loadBuckets();
      if (target === 'network') return loadNetwork();
      if (target === 'plan') return loadPlans();
      if (target === 'letters') return loadLetters();
      if (target === 'settings') return window.loadConfig();
      return Promise.resolve();
    };
    var refreshButton = document.getElementById('ob-refresh-dashboard');
    var refreshToast = document.createElement('div');
    refreshToast.className = 'ob-refresh-toast';
    refreshToast.setAttribute('role', 'status');
    refreshToast.setAttribute('aria-live', 'polite');
    refreshToast.innerHTML = '<span class="ob-refresh-toast-dot" aria-hidden="true"></span><span class="ob-refresh-toast-text"></span>';
    document.body.appendChild(refreshToast);
    var refreshToastTimer;
    refreshButton.addEventListener('click', async function () {
      if (refreshButton.disabled || !canLeaveEdit('编辑内容尚未保存，放弃修改并刷新当前页面？')) return;
      var activeTab = document.querySelector('.tab.active') || document.querySelector('.tab[data-tab="list"]');
      closeDetail();
      refreshButton.disabled = true;
      refreshButton.setAttribute('aria-busy', 'true');
      refreshButton.querySelector('span').textContent = '刷新中…';
      try {
        await window.activateDashboardTab(activeTab);
        clearTimeout(refreshToastTimer);
        refreshToast.querySelector('.ob-refresh-toast-text').textContent = '已刷新';
        refreshToast.classList.remove('show');
        void refreshToast.offsetWidth;
        refreshToast.classList.add('show');
        refreshToastTimer = setTimeout(function () { refreshToast.classList.remove('show'); }, 1000);
      } catch (error) {
        feedback('刷新失败：' + error.message);
      } finally {
        refreshButton.disabled = false;
        refreshButton.removeAttribute('aria-busy');
        refreshButton.querySelector('span').textContent = '刷新';
      }
    });
    window.loadConfig = async function () {
      var el = document.getElementById('ob-plugin-summary');
      try {
        var response = await fetch('/api/config');
        var config = await response.json();
        if (!response.ok) throw new Error(config.error || '读取插件配置失败');
        el.textContent = '会话模式：' + config.scope_mode + '；向量检索：' + (config.embedding.enabled ? '已启用' : '未启用') + '；自动打标：' + (config.tagging_enabled ? '已启用' : '未启用');
      } catch (error) { el.textContent = error.message; }
    };
    nativeDetail = window.showDetail;
    window.showDetail = async function (id) {
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并打开记忆？')) return false;
      var ok = await nativeDetail(id);
      if (ok) customizeDetail();
      return ok;
    };
    var originalClose = window.closeDetail;
    window.closeDetail = function () {
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并关闭？')) return false;
      originalClose();
      return true;
    };
    ['bucketPin','bucketResolve'].forEach(function (key) {
      var original = window[key];
      window[key] = function (id) {
        if (!canLeaveEdit('编辑内容尚未保存，放弃修改并更改记忆状态？')) return;
        return original(id);
      };
    });
    window.searchBuckets = async function (query, preservePage) {
      var revision = ++searchRevision;
      var session = sessionRevision;
      var matches = _localBucketMatches(query);
      var ids = new Set(matches.map(function (bucket) { return bucket.id; }));
      try {
        var response = await fetch('/api/search?q=' + encodeURIComponent(query));
        if (response.ok) {
          var rows = await response.json();
          rows.forEach(function (bucket) { if (!ids.has(bucket.id)) { ids.add(bucket.id); matches.push(bucket); } });
        }
      } catch (_) { /* 离线时保留当前会话的本地匹配。 */ }
      if (revision !== searchRevision || session !== sessionRevision || document.getElementById('search-input').value.trim() !== query) return;
      renderBuckets(filterBuckets(matches), preservePage);
    };
    window.bucketSaveEdit = saveEdit;
    document.querySelector('#memory-delete-dialog > p').textContent = '将永久删除所选记忆的正文、标签及关联检索向量，此操作无法撤销；已有导出文件和备份不受影响。';
    var nativeOpenDelete = window.openPermanentDelete;
    window.openPermanentDelete = function (ids) {
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并打开永久删除确认？')) return;
      nativeOpenDelete(ids);
    };
    window.confirmPermanentDelete = async function () {
      if (saving || permanentDeleteBusy || !pendingPermanentDeleteIds.length) return;
      var ids = pendingPermanentDeleteIds.slice();
      var deletionSession = selectedSession;
      var button = document.getElementById('memory-delete-confirm');
      var cancel = document.getElementById('memory-delete-cancel');
      var error = document.getElementById('memory-delete-error');
      saving = permanentDeleteBusy = true;
      button.disabled = cancel.disabled = true;
      button.textContent = '正在删除…';
      error.textContent = '';
      try {
        var response = await fetch('/api/buckets/permanent-delete', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({ids:ids,confirm:true})
        });
        var result = await response.json();
        if (!response.ok) throw Error(result.error || '永久删除失败，请重新登录后重试');
        (result.deleted || []).forEach(function (id) { selectedBucketIds.delete(id); });
        document.getElementById('memory-delete-dialog').close();
        pendingPermanentDeleteIds = [];
        dirty = false;
        originalClose();
        await window.refreshAstrbotSessions();
        if (selectedSession !== deletionSession) {
          sessionRevision += 1; searchRevision += 1; bucketLoadGeneration += 1;
          selectedBucketIds.clear(); allBuckets = []; currentFilter = 'active';
          document.getElementById('search-input').value = '';
          document.dispatchEvent(new Event('ob-session-changed'));
          renderBuckets([]); updateStats(); showView('list');
        }
        await loadBuckets();
        feedback('已永久删除 ' + result.deleted.length + ' 条记忆。');
      } catch (exception) { error.textContent = exception.message; }
      finally {
        saving = permanentDeleteBusy = false;
        button.disabled = cancel.disabled = false;
        button.textContent = '确认永久删除';
      }
    };
    window.bucketArchive = async function (id) {
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并归档？')) return;
      if (!confirm('确认归档这条记忆？正文仍会保留。')) return;
      try {
        var response = await fetch('/api/bucket/' + encodeURIComponent(id) + '/archive', {method:'POST'});
        var result = await response.json();
        if (!response.ok) throw new Error(result.error || '归档失败');
        dirty = false; closeDetail(); await loadBuckets(); feedback('记忆已归档。');
      } catch (error) { feedback(error.message); }
    };
    window.runBucketBatch = async function (action) {
      if (!canLeaveEdit('编辑内容尚未保存，放弃修改并批量操作？')) return;
      var ids = Array.from(selectedBucketIds);
      if (!ids.length) { feedback('请先选择记忆。'); return; }
      if (!['resolve','archive'].includes(action)) { feedback('插件版暂不支持此批量操作。'); return; }
      if (!confirm('确认将所选 ' + ids.length + ' 条记忆' + (action === 'resolve' ? '沉底' : '归档') + '？')) return;
      try {
        var response = await fetch('/api/buckets/batch', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:ids,action:action})});
        var result = await response.json();
        if (!response.ok) throw new Error(result.error || '批量操作失败');
        dirty = false; selectedBucketIds.clear(); closeDetail(); await loadBuckets();
        feedback('已更新 ' + result.updated.length + ' 条记忆。');
      } catch (error) { feedback(error.message); }
    };
    window.canSwitchDashboardFrontend = function () {
      return canLeaveEdit('编辑内容尚未保存，放弃修改并切换界面？');
    };
    window.addEventListener('beforeunload', function (event) {
      if (dirty || saving || (window.hasObInlineDrafts && window.hasObInlineDrafts()) || (window.hasObDomainDrafts && window.hasObDomainDrafts()) || (window.isObDomainSaving && window.isObDomainSaving())) { event.preventDefault(); event.returnValue = ''; }
    });
    window.loadVersionBadge = async function () {
      document.title = 'Ombre Brain · AstrBot';
      var badge = document.querySelector('.version-badge');
      if (badge) badge.textContent = '插件 1.3.0';
    };
    window.loadVersionBadge();
    document.querySelectorAll('.tab').forEach(function (tab) {
      if (unsupported[tab.dataset.tab]) {
        tab.classList.add('ob-unsupported-tab');
        tab.title = unsupported[tab.dataset.tab];
      }
    });
    var mode = document.getElementById('network-mode');
    if (mode) { mode.value = 'concept'; Array.from(mode.options).forEach(function (o) { o.disabled = o.value !== 'concept'; }); }
    var recovery = document.getElementById('auth-login-form');
    if (recovery) recovery.querySelectorAll('[onclick*="Recovery"], [data-cover-action="showRecovery"]').forEach(function (el) { el.remove(); });
    var warning = document.getElementById('auth-setup-remote-warning');
    if (warning) warning.textContent = '请设置仪表盘访问密码。两套界面共用同一账户和登录状态。';
    document.getElementById('heartbeat-indicator').title = 'AstrBot 记忆插件状态';
    window.updateStats();
    window.installObDomains();
    // The journal layout provides its own search and filter controls.
  }
  window.installAstrbotObDashboard = install;
  document.addEventListener('DOMContentLoaded', install, {once:true});
})();
