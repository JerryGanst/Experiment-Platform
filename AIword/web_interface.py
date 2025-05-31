"""
垂直化提示词补全Web界面
提供现代化的用户交互界面，实现智能提示词补全功能
"""

from flask import Flask, render_template, request, jsonify
import json
import time
from completion_engine import CompletionEngine, CompletionSuggestion

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aiword_completion_secret'

# 初始化补全引擎
completion_engine = CompletionEngine()


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/completion', methods=['POST'])
def get_completion():
    """获取补全建议API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        pause_time = data.get('pause_time', 0)
        
        # 获取补全建议
        suggestions = completion_engine.detect_completion(text, pause_time)
        
        # 转换为JSON格式
        suggestions_data = []
        for suggestion in suggestions:
            suggestions_data.append({
                'text': suggestion.text,
                'template': suggestion.template,
                'category': suggestion.category,
                'confidence': suggestion.confidence,
                'trigger_type': suggestion.trigger_type,
                'description': suggestion.description
            })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions_data,
            'total': len(suggestions_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/complete_question', methods=['POST'])
def complete_question():
    """生成完整问句API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        selected_option = data.get('selected_option', '')
        template = data.get('template', '')
        trigger_type = data.get('trigger_type', '')
        
        # 生成完整问句
        complete_text = completion_engine.generate_complete_question(
            text, selected_option, template, trigger_type
        )
        
        # 更新用户历史
        completion_engine.update_user_history(text, selected_option)
        
        return jsonify({
            'success': True,
            'complete_question': complete_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/popular_completions')
def get_popular_completions():
    """获取热门补全选项API"""
    try:
        category = request.args.get('category')
        popular = completion_engine.get_popular_completions(category)
        
        return jsonify({
            'success': True,
            'popular_completions': popular
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/add_domain_term', methods=['POST'])
def add_domain_term():
    """添加领域术语API"""
    try:
        data = request.get_json()
        term = data.get('term', '')
        options = data.get('options', [])
        template = data.get('template', '')
        category = data.get('category', '')
        
        completion_engine.add_domain_term(term, options, template, category)
        
        return jsonify({
            'success': True,
            'message': f'成功添加领域术语: {term}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/categories')
def get_categories():
    """获取所有分类API"""
    try:
        categories = completion_engine.get_categories()
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/floating')
def floating():
    """悬浮窗页面"""
    return render_template('floating.html')


if __name__ == '__main__':
    print("🚀 启动 AIword Web 服务器...")
    print("📝 普通模式: http://localhost:5000")
    print("🎈 悬浮窗模式: http://localhost:5000/floating")
    print("💡 按 Ctrl+C 停止服务")
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    ) 