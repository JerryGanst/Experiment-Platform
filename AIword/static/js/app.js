/**
 * AIword - 垂直化提示词补全系统
 * 前端交互逻辑实现
 */

class AIWordApp {
    constructor() {
        this.inputElement = document.getElementById('questionInput');
        this.completionLayer = document.getElementById('completionLayer');
        this.completionOptions = document.getElementById('completionOptions');
        this.submitBtn = document.getElementById('submitBtn');
        this.charCount = document.getElementById('charCount');
        
        this.lastInputTime = 0;
        this.currentSuggestions = [];
        this.selectedIndex = -1; // 当前选中的补全选项索引
        this.keyboardMode = false; // 是否处于键盘导航模式
        
        this.settings = {
            triggerDelay: 300,  // 调整为300ms提高响应速度
            maxSuggestions: 6,
            autoComplete: true
        };
        
        this.stats = {
            totalQuestions: 0,
            completionUsed: 0,
            totalResponseTime: 0,
            requestCount: 0
        };
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadSettings();
        this.loadStats();
        this.loadPopularCompletions();
        this.loadCategories();
        this.updateStats();
    }
    
    bindEvents() {
        // 输入框事件
        this.inputElement.addEventListener('input', this.handleInput.bind(this));
        this.inputElement.addEventListener('keydown', this.handleKeydown.bind(this));
        this.inputElement.addEventListener('blur', this.handleBlur.bind(this));
        this.inputElement.addEventListener('focus', this.handleFocus.bind(this));
        
        // 提交按钮
        this.submitBtn.addEventListener('click', this.handleSubmit.bind(this));
        
        // 补全层关闭按钮
        document.getElementById('closeCompletion').addEventListener('click', 
            this.hideCompletion.bind(this));
        
        // 全局点击事件（关闭补全层）
        document.addEventListener('click', (e) => {
            if (!this.completionLayer.contains(e.target) && 
                !this.inputElement.contains(e.target)) {
                this.hideCompletion();
            }
        });
        
        // 快捷键支持
        document.addEventListener('keydown', this.handleGlobalKeydown.bind(this));
    }
    
    handleInput(e) {
        const text = e.target.value;
        
        // 更新字符计数
        this.charCount.textContent = text.length;
        
        // 记录输入时间
        this.lastInputTime = Date.now();
        
        // 重置选择状态
        this.selectedIndex = -1;
        this.keyboardMode = false;
        
        // 延迟触发补全检测
        setTimeout(() => {
            if (Date.now() - this.lastInputTime >= this.settings.triggerDelay) {
                this.detectCompletion(text);
            }
        }, this.settings.triggerDelay);
    }
    
    handleKeydown(e) {
        const isCompletionVisible = !this.completionLayer.classList.contains('hidden');
        
        // Tab键触发补全
        if (e.key === 'Tab' && !isCompletionVisible) {
            e.preventDefault();
            this.triggerManualCompletion();
            return;
        }
        
        // ESC键关闭补全
        if (e.key === 'Escape') {
            this.hideCompletion();
            return;
        }
        
        // 处理补全层可见时的键盘操作
        if (isCompletionVisible) {
            this.handleCompletionKeydown(e);
            return;
        }
        
        // Enter键提交（Ctrl+Enter换行）
        if (e.key === 'Enter' && !e.ctrlKey) {
            e.preventDefault();
            this.handleSubmit();
        }
    }
    
    handleCompletionKeydown(e) {
        const options = this.completionOptions.querySelectorAll('.completion-option');
        
        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.navigateDown(options);
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                this.navigateUp(options);
                break;
                
            case 'Enter':
                e.preventDefault();
                this.selectCurrentOption();
                break;
                
            case 'Tab':
                e.preventDefault();
                this.selectCurrentOption();
                break;
                
            case 'Escape':
                e.preventDefault();
                this.hideCompletion();
                break;
                
            default:
                // 其他键继续正常输入，但保持补全层打开
                break;
        }
    }
    
    navigateDown(options) {
        if (options.length === 0) return;
        
        this.keyboardMode = true;
        
        // 清除当前高亮
        this.clearSelection();
        
        // 移动到下一项
        this.selectedIndex = (this.selectedIndex + 1) % options.length;
        
        // 高亮新选项
        this.highlightOption(options[this.selectedIndex]);
    }
    
    navigateUp(options) {
        if (options.length === 0) return;
        
        this.keyboardMode = true;
        
        // 清除当前高亮
        this.clearSelection();
        
        // 移动到上一项
        this.selectedIndex = this.selectedIndex <= 0 
            ? options.length - 1 
            : this.selectedIndex - 1;
        
        // 高亮新选项
        this.highlightOption(options[this.selectedIndex]);
    }
    
    clearSelection() {
        const options = this.completionOptions.querySelectorAll('.completion-option');
        options.forEach(option => {
            option.classList.remove('keyboard-selected');
        });
    }
    
    highlightOption(option) {
        if (option) {
            option.classList.add('keyboard-selected');
            // 确保选中项在视图中
            option.scrollIntoView({ block: 'nearest' });
        }
    }
    
    selectCurrentOption() {
        if (this.selectedIndex >= 0 && this.currentSuggestions[this.selectedIndex]) {
            const suggestion = this.currentSuggestions[this.selectedIndex];
            this.selectCompletion(suggestion, this.selectedIndex);
        }
    }
    
    triggerManualCompletion() {
        const text = this.inputElement.value;
        if (text.trim().length >= 1) {
            this.detectCompletion(text, 1000); // 强制触发，设置高pause_time
        }
    }
    
    handleFocus() {
        // 聚焦时显示提示
        this.showCompletionHint();
    }
    
    handleBlur() {
        // 延迟隐藏补全层，防止点击选项时立即关闭
        setTimeout(() => {
            if (!this.completionLayer.matches(':hover') && !this.keyboardMode) {
                this.hideCompletion();
            }
        }, 200);
    }
    
    handleGlobalKeydown(e) {
        // Ctrl+/ 显示帮助
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            this.showHelp();
        }
        
        // Ctrl+, 显示设置
        if (e.ctrlKey && e.key === ',') {
            e.preventDefault();
            this.showSettings();
        }
        
        // Ctrl+Space 强制触发补全
        if (e.ctrlKey && e.key === ' ') {
            e.preventDefault();
            this.triggerManualCompletion();
        }
    }
    
    async detectCompletion(text, forcePauseTime = null) {
        if (!text.trim() || text.length < 2) {
            this.hideCompletion();
            return;
        }
        
        try {
            const startTime = Date.now();
            const pauseTime = forcePauseTime !== null ? forcePauseTime : Date.now() - this.lastInputTime;
            
            const response = await fetch('/api/completion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    pause_time: pauseTime
                })
            });
            
            const data = await response.json();
            const responseTime = Date.now() - startTime;
            
            this.updateResponseTime(responseTime);
            
            if (data.success && data.suggestions.length > 0) {
                this.currentSuggestions = data.suggestions;
                this.showCompletion(data.suggestions);
            } else {
                this.hideCompletion();
            }
            
        } catch (error) {
            console.error('补全检测失败:', error);
            this.hideCompletion();
        }
    }
    
    showCompletion(suggestions) {
        this.completionOptions.innerHTML = '';
        this.selectedIndex = -1; // 重置选择索引
        
        suggestions.forEach((suggestion, index) => {
            const option = document.createElement('div');
            option.className = 'completion-option';
            option.innerHTML = `
                <div class="option-content">
                    <span class="option-text">${suggestion.text}</span>
                    <small class="option-category">(${suggestion.category})</small>
                </div>
                ${suggestion.description ? `<div class="option-description">${suggestion.description}</div>` : ''}
            `;
            
            // 鼠标事件
            option.addEventListener('click', () => {
                this.selectCompletion(suggestion, index);
            });
            
            option.addEventListener('mouseenter', () => {
                if (!this.keyboardMode) {
                    this.clearSelection();
                    this.selectedIndex = index;
                    this.highlightOption(option);
                }
            });
            
            option.addEventListener('mouseleave', () => {
                if (!this.keyboardMode) {
                    this.clearSelection();
                    this.selectedIndex = -1;
                }
            });
            
            this.completionOptions.appendChild(option);
        });
        
        this.completionLayer.classList.remove('hidden');
        
        // 添加显示动画
        this.completionLayer.style.animation = 'slideDown 0.3s ease-out';
        
        // 显示键盘提示
        this.showKeyboardHints();
    }
    
    hideCompletion() {
        this.completionLayer.classList.add('hidden');
        this.currentSuggestions = [];
        this.selectedIndex = -1;
        this.keyboardMode = false;
        this.hideKeyboardHints();
    }
    
    showKeyboardHints() {
        // 在补全层底部显示键盘操作提示
        const existingHint = this.completionLayer.querySelector('.keyboard-hints');
        if (!existingHint && this.currentSuggestions.length > 0) {
            const hintsDiv = document.createElement('div');
            hintsDiv.className = 'keyboard-hints';
            hintsDiv.innerHTML = `
                <small>
                    <kbd>↑</kbd><kbd>↓</kbd> 选择 | 
                    <kbd>Enter</kbd> 确认 | 
                    <kbd>Tab</kbd> 确认 | 
                    <kbd>Esc</kbd> 关闭
                </small>
            `;
            this.completionLayer.appendChild(hintsDiv);
        }
    }
    
    hideKeyboardHints() {
        const hint = this.completionLayer.querySelector('.keyboard-hints');
        if (hint) {
            hint.remove();
        }
    }
    
    showCompletionHint() {
        // 在输入框获得焦点时显示提示
        if (this.inputElement.value.length === 0) {
            this.showNotification('按 Tab 键或 Ctrl+Space 触发智能补全', 'info');
        }
    }
    
    async selectCompletion(suggestion, index) {
        const currentText = this.inputElement.value;
        
        try {
            const response = await fetch('/api/complete_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: currentText,
                    selected_option: suggestion.text.replace(/[🔹📊]/g, '').trim(),
                    template: suggestion.template,
                    trigger_type: suggestion.trigger_type
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.inputElement.value = data.complete_question;
                this.charCount.textContent = data.complete_question.length;
                this.hideCompletion();
                
                // 更新统计
                this.stats.completionUsed++;
                this.updateStats();
                
                // 显示成功提示
                this.showNotification('补全成功！', 'success');
                
                // 聚焦到输入框末尾
                this.inputElement.focus();
                this.inputElement.setSelectionRange(
                    this.inputElement.value.length, 
                    this.inputElement.value.length
                );
            }
            
        } catch (error) {
            console.error('补全生成失败:', error);
            this.showNotification('补全失败，请重试', 'error');
        }
    }
    
    navigateCompletions(down) {
        const options = this.completionOptions.querySelectorAll('.completion-option');
        const current = this.completionOptions.querySelector('.completion-option.active');
        
        let newIndex = 0;
        
        if (current) {
            const currentIndex = Array.from(options).indexOf(current);
            current.classList.remove('active');
            
            if (down) {
                newIndex = (currentIndex + 1) % options.length;
            } else {
                newIndex = (currentIndex - 1 + options.length) % options.length;
            }
        }
        
        if (options[newIndex]) {
            options[newIndex].classList.add('active');
            
            // Enter键选择当前高亮项
            const enterHandler = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    options[newIndex].click();
                    document.removeEventListener('keydown', enterHandler);
                }
            };
            
            document.addEventListener('keydown', enterHandler);
        }
    }
    
    handleSubmit() {
        const text = this.inputElement.value.trim();
        
        if (!text) {
            this.showNotification('请输入问题内容', 'warning');
            return;
        }
        
        // 更新统计
        this.stats.totalQuestions++;
        this.updateStats();
        this.saveStats();
        
        // 这里可以添加提交到AI服务的逻辑
        this.showNotification('问题已提交处理', 'success');
        
        // 清空输入框（可选）
        // this.inputElement.value = '';
        // this.charCount.textContent = '0';
    }
    
    async loadPopularCompletions() {
        try {
            const response = await fetch('/api/popular_completions');
            const data = await response.json();
            
            if (data.success) {
                this.renderPopularTags(data.popular_completions);
            }
        } catch (error) {
            console.error('加载热门补全失败:', error);
        }
    }
    
    renderPopularTags(completions) {
        const container = document.getElementById('popularCompletions');
        container.innerHTML = '';
        
        completions.forEach(completion => {
            const tag = document.createElement('div');
            tag.className = 'popular-tag';
            tag.textContent = completion;
            
            tag.addEventListener('click', () => {
                const currentText = this.inputElement.value;
                const newText = currentText ? `${currentText} ${completion}` : completion;
                this.inputElement.value = newText;
                this.charCount.textContent = newText.length;
                this.inputElement.focus();
            });
            
            container.appendChild(tag);
        });
    }
    
    async loadCategories() {
        try {
            const response = await fetch('/api/categories');
            const data = await response.json();
            
            if (data.success) {
                this.renderCategories(data.categories);
            }
        } catch (error) {
            console.error('加载分类失败:', error);
        }
    }
    
    renderCategories(categories) {
        const container = document.getElementById('categoriesGrid');
        container.innerHTML = '';
        
        const iconMap = {
            'AI/ML': 'fas fa-robot',
            'Data Science': 'fas fa-chart-bar',
            'Development': 'fas fa-code',
            'Emerging Tech': 'fas fa-rocket',
            '编程语言': 'fas fa-code',
            '数据科学': 'fas fa-chart-bar',
            '新兴技术': 'fas fa-rocket'
        };
        
        Object.entries(categories).forEach(([key, category]) => {
            const card = document.createElement('div');
            card.className = 'category-card';
            card.innerHTML = `
                <i class="category-icon ${iconMap[key] || 'fas fa-folder'}"></i>
                <div class="category-name">${category.name}</div>
                <div class="category-count">${category.common_patterns.length} 个术语</div>
            `;
            
            card.addEventListener('click', () => {
                this.filterByCategory(key);
            });
            
            container.appendChild(card);
        });
    }
    
    filterByCategory(category) {
        // 这里可以实现按分类筛选功能
        this.showNotification(`已选择 ${category} 分类`, 'info');
    }
    
    updateStats() {
        document.getElementById('totalQuestions').textContent = this.stats.totalQuestions;
        
        const completionRate = this.stats.totalQuestions > 0 
            ? Math.round((this.stats.completionUsed / this.stats.totalQuestions) * 100)
            : 0;
        document.getElementById('completionRate').textContent = `${completionRate}%`;
        
        const avgResponseTime = this.stats.requestCount > 0
            ? Math.round(this.stats.totalResponseTime / this.stats.requestCount)
            : 0;
        document.getElementById('avgResponseTime').textContent = `${avgResponseTime}ms`;
    }
    
    updateResponseTime(time) {
        this.stats.totalResponseTime += time;
        this.stats.requestCount++;
        this.updateStats();
    }
    
    loadSettings() {
        const saved = localStorage.getItem('aiword_settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }
    
    saveSettings() {
        localStorage.setItem('aiword_settings', JSON.stringify(this.settings));
    }
    
    loadStats() {
        const saved = localStorage.getItem('aiword_stats');
        if (saved) {
            this.stats = { ...this.stats, ...JSON.parse(saved) };
        }
    }
    
    saveStats() {
        localStorage.setItem('aiword_stats', JSON.stringify(this.stats));
    }
    
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const icon = document.getElementById('notificationIcon');
        const text = document.getElementById('notificationText');
        
        // 设置图标和样式
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        const colorMap = {
            success: '#4CAF50',
            error: '#f44336',
            warning: '#FF9800',
            info: '#2196F3'
        };
        
        icon.className = iconMap[type] || iconMap.success;
        text.textContent = message;
        notification.style.background = colorMap[type] || colorMap.success;
        
        notification.classList.remove('hidden');
        
        // 自动隐藏
        setTimeout(() => {
            notification.classList.add('hidden');
        }, 3000);
    }
    
    showSettings() {
        const modal = document.getElementById('settingsModal');
        
        // 加载当前设置
        document.getElementById('triggerDelay').value = this.settings.triggerDelay;
        document.getElementById('maxSuggestions').value = this.settings.maxSuggestions;
        document.getElementById('autoComplete').checked = this.settings.autoComplete;
        
        modal.classList.remove('hidden');
    }
    
    showHelp() {
        this.showNotification('快捷键：Tab 触发补全，↑↓ 选择，Enter 确认，Esc 关闭', 'info');
    }
    
    showAbout() {
        this.showNotification('AIword v1.0 - 垂直化提示词补全系统', 'info');
    }
}

// 全局函数（供HTML调用）
function hideModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

function showSettings() {
    window.aiwordApp.showSettings();
}

function showHelp() {
    window.aiwordApp.showHelp();
}

function showAbout() {
    window.aiwordApp.showAbout();
}

function saveSettings() {
    const app = window.aiwordApp;
    
    app.settings.triggerDelay = parseInt(document.getElementById('triggerDelay').value);
    app.settings.maxSuggestions = parseInt(document.getElementById('maxSuggestions').value);
    app.settings.autoComplete = document.getElementById('autoComplete').checked;
    
    app.saveSettings();
    hideModal('settingsModal');
    app.showNotification('设置已保存', 'success');
}

async function addDomainTerm() {
    const app = window.aiwordApp;
    
    const term = document.getElementById('termName').value.trim();
    const options = document.getElementById('termOptions').value.trim().split(',');
    const template = document.getElementById('termTemplate').value.trim();
    const category = document.getElementById('termCategory').value;
    
    if (!term || !options.length || !template) {
        app.showNotification('请填写完整信息', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/add_domain_term', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                term: term,
                options: options.map(opt => opt.trim()),
                template: template,
                category: category
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            app.showNotification(data.message, 'success');
            hideModal('addTermModal');
            
            // 清空表单
            document.getElementById('termName').value = '';
            document.getElementById('termOptions').value = '';
            document.getElementById('termTemplate').value = '';
        } else {
            app.showNotification(data.error || '添加失败', 'error');
        }
        
    } catch (error) {
        console.error('添加术语失败:', error);
        app.showNotification('网络错误，请重试', 'error');
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.aiwordApp = new AIWordApp();
}); 