"""
股票预测系统 - Android Kivy版本
使用Kivy框架创建Android应用界面
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
import threading

# 设置窗口大小（开发时使用）
Window.size = (360, 640)

# 导入预测模块
try:
    from stock_live_prediction_APP import (
        load_trained_model,
        load_all_models,
        load_feature_list,
        load_model_info,
        download_stock_for_prediction,
        extract_features_for_prediction,
        align_features,
        predict_stock,
        predict_with_all_models
    )
    PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 预测模块导入失败: {e}")
    PREDICTION_AVAILABLE = False


class StockPredictorApp(App):
    """股票预测应用主类"""
    
    def build(self):
        """构建UI界面"""
        self.title = "股票预测系统"
        
        # 主布局
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # 标题
        title = Label(
            text='📊 股票预测系统',
            size_hint=(1, 0.1),
            font_size='24sp',
            bold=True
        )
        layout.add_widget(title)
        
        # 股票代码输入
        input_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.08), spacing=5)
        input_label = Label(text='股票代码:', size_hint=(0.3, 1))
        self.stock_input = TextInput(
            hint_text='输入6位数字',
            multiline=False,
            size_hint=(0.7, 1),
            input_filter='int'
        )
        input_layout.add_widget(input_label)
        input_layout.add_widget(self.stock_input)
        layout.add_widget(input_layout)
        
        # 预测模式选择
        mode_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.08), spacing=5)
        mode_label = Label(text='预测模式:', size_hint=(0.3, 1))
        self.mode_spinner = Spinner(
            text='单模型',
            values=('单模型', '多模型对比'),
            size_hint=(0.7, 1)
        )
        mode_layout.add_widget(mode_label)
        mode_layout.add_widget(self.mode_spinner)
        layout.add_widget(mode_layout)
        
        # 快速选择按钮
        quick_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.08), spacing=5)
        btn_moutai = Button(text='贵州茅台', on_press=lambda x: self.quick_select('600519'))
        btn_pingan = Button(text='平安银行', on_press=lambda x: self.quick_select('000001'))
        btn_zhaoshang = Button(text='招商银行', on_press=lambda x: self.quick_select('600036'))
        quick_layout.add_widget(btn_moutai)
        quick_layout.add_widget(btn_pingan)
        quick_layout.add_widget(btn_zhaoshang)
        layout.add_widget(quick_layout)
        
        # 预测按钮
        self.predict_btn = Button(
            text='🔮 开始预测',
            size_hint=(1, 0.1),
            background_color=(0.2, 0.6, 1, 1),
            on_press=self.start_prediction
        )
        layout.add_widget(self.predict_btn)
        
        # 进度条
        self.progress = ProgressBar(max=100, size_hint=(1, 0.03))
        layout.add_widget(self.progress)
        
        # 结果显示区域（可滚动）
        scroll = ScrollView(size_hint=(1, 0.53))
        self.result_label = Label(
            text='请输入股票代码并点击预测',
            size_hint_y=None,
            text_size=(Window.width - 40, None),
            halign='left',
            valign='top',
            markup=True
        )
        self.result_label.bind(texture_size=self.result_label.setter('size'))
        scroll.add_widget(self.result_label)
        layout.add_widget(scroll)
        
        return layout
    
    def quick_select(self, code):
        """快速选择股票"""
        self.stock_input.text = code
    
    def start_prediction(self, instance):
        """开始预测（在后台线程）"""
        stock_code = self.stock_input.text.strip()
        
        if not stock_code:
            self.show_result("[color=ff0000]❌ 错误：请输入股票代码[/color]")
            return
        
        if len(stock_code) != 6 or not stock_code.isdigit():
            self.show_result("[color=ff0000]❌ 错误：股票代码必须是6位数字[/color]")
            return
        
        # 禁用按钮
        self.predict_btn.disabled = True
        self.predict_btn.text = '预测中...'
        self.show_result("[color=0080ff]📥 正在下载数据...[/color]")
        
        # 重置进度条
        self.progress.value = 0
        
        # 在后台线程执行预测
        threading.Thread(
            target=self.predict_stock,
            args=(stock_code,),
            daemon=True
        ).start()
    
    def predict_stock(self, stock_code):
        """执行预测（后台线程）"""
        try:
            if not PREDICTION_AVAILABLE:
                self.update_ui(
                    "[color=ff0000]❌ 预测模块不可用[/color]\n"
                    "请确保已正确安装所有依赖"
                )
                return
            
            # 更新进度：10%
            Clock.schedule_once(lambda dt: self.update_progress(10), 0)
            
            # 1. 加载模型
            use_multi = (self.mode_spinner.text == '多模型对比')
            
            if use_multi:
                all_models = load_all_models()
                if all_models is None:
                    model = load_trained_model()
                    use_multi = False
                else:
                    model = None
            else:
                model = load_trained_model()
                all_models = None
            
            feature_list = load_feature_list()
            model_info = load_model_info()
            
            if (model is None and all_models is None) or feature_list is None:
                self.update_ui("[color=ff0000]❌ 模型文件加载失败[/color]")
                return
            
            # 更新进度：30%
            Clock.schedule_once(lambda dt: self.update_progress(30), 0)
            
            # 2. 下载数据
            stock_data = download_stock_for_prediction(stock_code, days=365)
            if stock_data is None:
                self.update_ui(f"[color=ff0000]❌ 无法下载股票 {stock_code} 的数据[/color]")
                return
            
            # 更新进度：50%
            Clock.schedule_once(lambda dt: self.update_progress(50), 0)
            
            # 3. 提取特征
            features_df = extract_features_for_prediction(stock_data, window_size=20)
            if features_df is None:
                self.update_ui("[color=ff0000]❌ 特征提取失败[/color]")
                return
            
            # 更新进度：70%
            Clock.schedule_once(lambda dt: self.update_progress(70), 0)
            
            # 4. 对齐特征
            aligned_features = align_features(features_df, feature_list)
            
            # 更新进度：80%
            Clock.schedule_once(lambda dt: self.update_progress(80), 0)
            
            # 5. 预测
            if use_multi and all_models:
                predictions_dict = predict_with_all_models(all_models, aligned_features)
                result_text = self.format_multi_model_result(
                    stock_code, stock_data, predictions_dict, model_info
                )
            else:
                prediction, probability = predict_stock(model, aligned_features)
                result_text = self.format_single_result(
                    stock_code, stock_data, prediction, probability, model_info
                )
            
            # 更新进度：100%
            Clock.schedule_once(lambda dt: self.update_progress(100), 0)
            
            # 显示结果
            self.update_ui(result_text)
            
        except Exception as e:
            import traceback
            error_msg = f"[color=ff0000]❌ 预测失败：{str(e)}[/color]\n\n{traceback.format_exc()}"
            self.update_ui(error_msg)
    
    def format_single_result(self, stock_code, stock_data, prediction, probability, model_info):
        """格式化单模型预测结果"""
        latest = stock_data.iloc[-1]
        latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
        
        # 计算MA20
        if 'MA_20' in stock_data.columns:
            ma20 = latest['MA_20']
        elif 'SMA_20' in stock_data.columns:
            ma20 = latest['SMA_20']
        else:
            ma20 = stock_data['Close'].tail(20).mean()
        
        current_price = latest['Close']
        position_diff = ((current_price - ma20) / ma20) * 100
        
        # 预测结果
        prediction_label = "强势" if prediction == 0 else "弱势"
        prob_strong = probability[0]
        prob_weak = probability[1]
        confidence = max(prob_strong, prob_weak)
        
        # 颜色
        pred_color = "00ff00" if prediction == 0 else "ff8800"
        
        result = f"""[b][size=18sp]📊 {stock_code} 预测结果[/size][/b]

[b]📅 最新数据[/b]
日期: {latest_date}
收盘价: [color=0080ff]{current_price:.2f}[/color] 元
MA20: {ma20:.2f} 元
当前: {'强势' if current_price >= ma20 else '弱势'} ({position_diff:+.2f}%)

[b]🔮 预测结果（未来5日）[/b]
预测: [color={pred_color}]{prediction_label}[/color]
强势概率: {prob_strong:.1%}
弱势概率: {prob_weak:.1%}
置信度: [color=0080ff]{confidence:.1%}[/color]

[b]💡 操作建议[/b]
"""
        
        if prediction == 0:
            if prob_strong >= 0.75:
                result += "✅ 可考虑持有或增仓\n预测强势且置信度高"
            else:
                result += "📊 观察为主，谨慎持有\n预测强势但置信度中等"
        else:
            if prob_weak >= 0.75:
                result += "⚠️ 考虑减仓或观望\n预测弱势且置信度高"
            else:
                result += "📊 谨慎持有，控制仓位\n预测弱势但置信度中等"
        
        result += "\n\n[color=ff0000]⚠️ 风险提示[/color]\n"
        result += "预测仅供参考，不构成投资建议\n"
        result += "股票投资有风险，入市需谨慎"
        
        return result
    
    def format_multi_model_result(self, stock_code, stock_data, predictions_dict, model_info):
        """格式化多模型预测结果"""
        latest = stock_data.iloc[-1]
        latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
        
        # 计算MA20
        if 'MA_20' in stock_data.columns:
            ma20 = latest['MA_20']
        elif 'SMA_20' in stock_data.columns:
            ma20 = latest['SMA_20']
        else:
            ma20 = stock_data['Close'].tail(20).mean()
        
        current_price = latest['Close']
        position_diff = ((current_price - ma20) / ma20) * 100
        
        # 投票统计
        vote_strong = sum(1 for p in predictions_dict.values() if p['prediction'] == 0)
        vote_weak = len(predictions_dict) - vote_strong
        
        result = f"""[b][size=18sp]📊 {stock_code} 多模型预测[/size][/b]

[b]📅 最新数据[/b]
日期: {latest_date}
收盘价: [color=0080ff]{current_price:.2f}[/color] 元
MA20: {ma20:.2f} 元

[b]🔮 投票结果[/b]
"""
        
        if vote_strong > vote_weak:
            result += f"[color=00ff00]✅ 多数预测: 强势[/color]\n"
        else:
            result += f"[color=ff8800]⚠️ 多数预测: 弱势[/color]\n"
        
        result += f"强势: {vote_strong}/{len(predictions_dict)}\n"
        result += f"弱势: {vote_weak}/{len(predictions_dict)}\n\n"
        
        result += "[b]📈 各模型预测[/b]\n"
        
        for model_name, pred in sorted(
            predictions_dict.items(), 
            key=lambda x: x[1]['confidence'], 
            reverse=True
        ):
            pred_label = "强势" if pred['prediction'] == 0 else "弱势"
            color = "00ff00" if pred['prediction'] == 0 else "ff8800"
            result += f"[color={color}]{model_name}[/color]: {pred_label} ({pred['confidence']:.1%})\n"
        
        # 平均置信度
        avg_conf = sum(p['confidence'] for p in predictions_dict.values()) / len(predictions_dict)
        result += f"\n平均置信度: [color=0080ff]{avg_conf:.1%}[/color]\n"
        
        result += "\n[color=ff0000]⚠️ 风险提示[/color]\n"
        result += "预测仅供参考，不构成投资建议"
        
        return result
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress.value = value
    
    def update_ui(self, text):
        """更新UI（从主线程调用）"""
        def _update(dt):
            self.result_label.text = text
            self.predict_btn.disabled = False
            self.predict_btn.text = '🔮 开始预测'
            self.progress.value = 0
        
        Clock.schedule_once(_update, 0)
    
    def show_result(self, text):
        """显示结果"""
        self.result_label.text = text


def main():
    """主函数"""
    try:
        StockPredictorApp().run()
    except Exception as e:
        print(f"应用启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


