/**
 * AstrBot Memory Dashboard Controller - Vue 3 Single Page Application
 */

// Helper: Hex color to RGB
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

// Helper: Adjust color brightness (darken/lighten)
function darkenColor(hex, percent) {
  const num = parseInt(hex.replace("#", ""), 16);
  const amt = Math.round(2.55 * percent);
  let R = (num >> 16) + amt;
  let G = (num >> 8 & 0x00FF) + amt;
  let B = (num & 0x0000FF) + amt;
  R = Math.max(0, Math.min(255, R));
  G = Math.max(0, Math.min(255, G));
  B = Math.max(0, Math.min(255, B));
  return "#" + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
}


// Instantiate Vue App
const { createApp } = Vue;

createApp({
  data() {
    return {
      // Auth states
      setupNeeded: false,
      isAuthenticated: false,
      isLoading: false,
      password: '',
      confirmPassword: '',
      currentPassword: '',
      newPassword: '',

      // View configurations
      currentTab: 'overview',
      activeTheme: '',
      customColor: '#a78bda',

      // Data states
      stats: null,
      buckets: [],
      sessions: [],
      selectedSession: '',
      selectedType: 'dynamic',

      // Search state
      searchQuery: '',
      searchResults: [],
      searchDone: false,
      isSearching: false,

      // Inline Editor states
      editingBucketId: null,
      editForm: {
        name: '',
        content: '',
        domain: [],
        tags: [],
        valence: 0.5,
        arousal: 0.3,
        importance: 5,
        pinned: false,
        resolved: false,
        digested: false
      },

      // Toast feedback banner states
      toastMsg: '',
      toastTimeout: null
    };
  },

  computed: {
    // Number of pinned (permanent) memories
    pinnedCount() {
      return this.buckets.filter(b => b.pinned || b.bucket_type === 'permanent').length;
    },

    // Filtered buckets for the Kanban card display list
    filteredBuckets() {
      return this.buckets.filter(b => {
        // Filter by Session
        if (this.selectedSession && b.session_id !== this.selectedSession) {
          return false;
        }
        // Filter by Bucket Type classification
        return b.bucket_type === this.selectedType;
      });
    },

    // Scatter plot dots for circumplex emotional plane
    emotionDots() {
      // Exclude archived/resolved buckets from displaying on emotion plane to avoid clutter
      return this.buckets.filter(b => b.bucket_type !== 'archived' && !b.resolved);
    },

    // Selected bucket for emotion plot inline editor
    activeEmotionBucket() {
      return this.buckets.find(b => b.id === this.editingBucketId);
    }
  },

  mounted() {
    // Check initial authentication and configuration status
    this.checkAuthStatus();
    
    // Restore styling preferences from local storage
    this.restoreTheme();
  },

  methods: {
    // --- Toast alerts ---
    showToast(message) {
      if (this.toastTimeout) {
        clearTimeout(this.toastTimeout);
      }
      this.toastMsg = message;
      this.toastTimeout = setTimeout(() => {
        this.toastMsg = '';
      }, 3500);
    },

    // --- API Service Calls ---
    async apiRequest(url, options = {}) {
      try {
        const response = await fetch(url, {
          headers: {
            'Content-Type': 'application/json',
            ...options.headers
          },
          ...options
        });
        
        if (response.status === 401 && this.isAuthenticated) {
          this.isAuthenticated = false;
          this.showToast('会话已过期，请重新登录');
          return null;
        }

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || data.detail || '请求失败');
        }
        return data;
      } catch (err) {
        this.showToast(err.message);
        console.error('API Error:', err);
        return null;
      }
    },

    async checkAuthStatus() {
      this.isLoading = true;
      const status = await this.apiRequest('/auth/status');
      this.isLoading = false;
      
      if (status) {
        this.setupNeeded = status.setup_needed;
        this.isAuthenticated = status.authenticated;
        
        // Fetch dashboard statistics and buckets list immediately if logged in
        if (this.isAuthenticated) {
          this.fetchStats();
          this.loadBuckets();
        }
      }
    },

    async handleSetup() {
      if (this.password.length < 4) {
        this.showToast('密码长度必须至少为 4 位');
        return;
      }
      if (this.password !== this.confirmPassword) {
        this.showToast('两次输入的密码不一致');
        return;
      }

      this.isLoading = true;
      const res = await this.apiRequest('/auth/setup', {
        method: 'POST',
        body: JSON.stringify({ password: this.password })
      });
      this.isLoading = false;

      if (res && res.ok) {
        this.showToast('密码配置成功并自动登录');
        this.isAuthenticated = true;
        this.setupNeeded = false;
        this.password = '';
        this.confirmPassword = '';
        this.fetchStats();
        this.loadBuckets();
      }
    },

    async handleLogin() {
      this.isLoading = true;
      const res = await this.apiRequest('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ password: this.password })
      });
      this.isLoading = false;

      if (res && res.ok) {
        this.showToast('登录成功');
        this.isAuthenticated = true;
        this.password = '';
        this.fetchStats();
        this.loadBuckets();
      }
    },

    async handleLogout() {
      const res = await this.apiRequest('/auth/logout', { method: 'POST' });
      if (res && res.ok) {
        this.showToast('已安全退出登录');
        this.isAuthenticated = false;
        this.buckets = [];
        this.sessions = [];
        this.stats = null;
      }
    },

    async handleChangePassword() {
      if (this.newPassword.length < 4) {
        this.showToast('新密码长度必须至少为 4 位');
        return;
      }

      this.isLoading = true;
      const res = await this.apiRequest('/auth/change-password', {
        method: 'POST',
        body: JSON.stringify({
          current: this.currentPassword,
          new: this.newPassword
        })
      });
      this.isLoading = false;

      if (res && res.ok) {
        this.showToast('密码修改成功');
        this.currentPassword = '';
        this.newPassword = '';
      }
    },

    async fetchStats() {
      const data = await this.apiRequest('/api/stats');
      if (data) {
        this.stats = data;
      }
    },

    async loadBuckets() {
      this.isLoading = true;
      // Fetch dynamic, permanent, feel, and archived buckets in one go
      let url = '/api/buckets?limit=250';
      if (this.selectedSession) {
        url += `&session=${encodeURIComponent(this.selectedSession)}`;
      }
      
      const data = await this.apiRequest(url);
      this.isLoading = false;
      
      if (data && data.buckets) {
        this.buckets = data.buckets;
        
        // Extract unique session list dynamically
        const uniqueSessions = [...new Set(this.buckets.map(b => b.session_id))].filter(Boolean);
        this.sessions = uniqueSessions;
      }
    },

    async handleSearch() {
      if (!this.searchQuery.trim()) {
        this.showToast('请输入检索关键词');
        return;
      }

      this.isSearching = true;
      this.searchDone = false;
      
      let url = `/api/search?q=${encodeURIComponent(this.searchQuery.trim())}`;
      if (this.selectedSession) {
        url += `&session=${encodeURIComponent(this.selectedSession)}`;
      }

      const data = await this.apiRequest(url);
      this.isSearching = false;
      this.searchDone = true;

      if (data && data.results) {
        this.searchResults = data.results;
      }
    },

    // --- Inline Card Editor methods ---
    startEdit(bucket) {
      this.editingBucketId = bucket.id;
      this.editForm = {
        name: bucket.name || '',
        content: bucket.content || '',
        domain: Array.isArray(bucket.domain) ? [...bucket.domain] : [],
        tags: Array.isArray(bucket.tags) ? [...bucket.tags] : [],
        valence: typeof bucket.valence === 'number' ? bucket.valence : 0.5,
        arousal: typeof bucket.arousal === 'number' ? bucket.arousal : 0.3,
        importance: typeof bucket.importance === 'number' ? bucket.importance : 5,
        pinned: !!bucket.pinned,
        resolved: !!bucket.resolved,
        digested: !!bucket.digested
      };
    },

    cancelEdit() {
      this.editingBucketId = null;
    },

    async saveEdit() {
      if (!this.editingBucketId) return;

      this.isLoading = true;
      const res = await this.apiRequest(`/api/bucket/${this.editingBucketId}`, {
        method: 'PATCH',
        body: JSON.stringify(this.editForm)
      });
      this.isLoading = false;

      if (res) {
        this.showToast('更改已保存');
        this.cancelEdit();
        
        // Refresh local lists
        this.loadBuckets();
        this.fetchStats();
      }
    },

    async deleteEdit() {
      if (!this.editingBucketId) return;
      if (!confirm('确定要彻底删除该长效记忆块吗？删除后不可恢复。')) return;

      this.isLoading = true;
      const res = await this.apiRequest(`/api/bucket/${this.editingBucketId}`, {
        method: 'DELETE'
      });
      this.isLoading = false;

      if (res) {
        this.showToast('记忆已成功移除');
        this.cancelEdit();
        
        // Refresh local lists
        this.loadBuckets();
        this.fetchStats();
      }
    },

    addEditTagOrDomain(event) {
      const val = event.target.value.trim();
      if (val) {
        if (val.startsWith('@')) {
          const domainName = val.slice(1).trim();
          if (domainName && !this.editForm.domain.includes(domainName)) {
            this.editForm.domain.push(domainName);
          }
        } else {
          if (!this.editForm.tags.includes(val)) {
            this.editForm.tags.push(val);
          }
        }
      }
      event.target.value = '';
    },

    removeEditTag(index) {
      this.editForm.tags.splice(index, 1);
    },

    removeEditDomain(index) {
      this.editForm.domain.splice(index, 1);
    },

    switchType(type) {
      this.selectedType = type;
      this.cancelEdit();
    },

    // --- Theme Config Functions ---
    restoreTheme() {
      const savedTheme = localStorage.getItem('memory_dashboard_theme');
      const savedColor = localStorage.getItem('memory_dashboard_custom_color');
      
      if (savedTheme) {
        this.activeTheme = savedTheme;
        if (savedTheme === 'custom' && savedColor) {
          this.customColor = savedColor;
          this.applyCustomThemeColor(savedColor);
        } else {
          document.documentElement.setAttribute('data-theme', savedTheme);
        }
      }
    },

    setTheme(themeName) {
      this.activeTheme = themeName;
      localStorage.setItem('memory_dashboard_theme', themeName);
      
      if (themeName !== 'custom') {
        // Clear style overrides and set attribute data-theme
        document.documentElement.removeAttribute('style');
        if (themeName) {
          document.documentElement.setAttribute('data-theme', themeName);
        } else {
          document.documentElement.removeAttribute('data-theme');
        }
      } else {
        this.applyCustomThemeColor(this.customColor);
      }
    },

    onCustomColorChange(event) {
      const color = event.target.value;
      this.customColor = color;
      this.activeTheme = 'custom';
      localStorage.setItem('memory_dashboard_theme', 'custom');
      localStorage.setItem('memory_dashboard_custom_color', color);
      this.applyCustomThemeColor(color);
    },

    applyCustomThemeColor(hex) {
      const root = document.documentElement;
      root.setAttribute('data-theme', 'custom');
      
      // Calculate dynamic theme styles using CSS custom variables
      root.style.setProperty('--primary', hex);
      const rgb = hexToRgb(hex);
      if (rgb) {
        root.style.setProperty('--primary-dark', darkenColor(hex, -15));
        root.style.setProperty('--primary-light', `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.15)`);
        root.style.setProperty('--primary-border', `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.25)`);
        root.style.setProperty('--primary-hover', `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.4)`);
        root.style.setProperty('--primary-glow', `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.35)`);
        
        // Ambient background gradient matching custom primary color (light base by default)
        root.style.setProperty('--bg-gradient-1', darkenColor(hex, 45));
        root.style.setProperty('--bg-gradient-2', '#faf8fd');
        root.style.setProperty('--bg-gradient-3', '#ffffff');
      }
    },

    // --- Emotion Circumplex Scatter Plot Math ---
    getDotStyle(bucket) {
      // Map Russell valence range [0.0, 1.0] to CSS left [0%, 100%]
      // Map Russell arousal range [0.0, 1.0] to CSS bottom [0%, 100%]
      const xPercent = bucket.valence * 100;
      const yPercent = bucket.arousal * 100;
      return {
        left: `${xPercent}%`,
        bottom: `${yPercent}%`
      };
    }
  }
}).mount('#app');

// PWA Service Worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
}
