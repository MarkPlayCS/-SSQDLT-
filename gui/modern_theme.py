"""
现代主题样式模块
提供现代化的UI主题和样式
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont


class ModernTheme:
    """现代主题类"""
    
    def __init__(self):
        self.colors = {
            # 主色调
            'primary': '#2E86AB',      # 深蓝色
            'primary_dark': '#1B4F72', # 深蓝色暗色
            'primary_light': '#85C1E9', # 浅蓝色
            
            # 辅助色
            'secondary': '#F39C12',    # 橙色
            'success': '#27AE60',      # 绿色
            'warning': '#F1C40F',      # 黄色
            'danger': '#E74C3C',       # 红色
            'info': '#3498DB',         # 信息蓝
            
            # 中性色
            'light': '#ECF0F1',        # 浅灰
            'dark': '#2C3E50',         # 深灰
            'white': '#FFFFFF',        # 白色
            'black': '#000000',        # 黑色
            
            # 背景色
            'bg_primary': '#FFFFFF',   # 主背景
            'bg_secondary': '#F8F9FA', # 次背景
            'bg_card': '#FFFFFF',      # 卡片背景
            'bg_sidebar': '#2C3E50',   # 侧边栏背景
            
            # 文字色
            'text_primary': '#2C3E50', # 主文字
            'text_secondary': '#7F8C8D', # 次文字
            'text_light': '#BDC3C7',   # 浅文字
            'text_white': '#FFFFFF',   # 白色文字
            
            # 边框色
            'border': '#E1E8ED',       # 边框
            'border_focus': '#2E86AB', # 焦点边框
        }
        
        self.fonts = {
            'title': ('Microsoft YaHei UI', 18, 'bold'),
            'heading': ('Microsoft YaHei UI', 14, 'bold'),
            'subheading': ('Microsoft YaHei UI', 12, 'bold'),
            'body': ('Microsoft YaHei UI', 10),
            'small': ('Microsoft YaHei UI', 9),
            'monospace': ('Consolas', 10),
        }
        
        self.spacing = {
            'xs': 4,
            'sm': 8,
            'md': 12,
            'lg': 16,
            'xl': 24,
            'xxl': 32,
        }
        
        self.border_radius = 8
        self.shadow_offset = 2
    
    def configure_style(self):
        """配置ttk样式"""
        style = ttk.Style()
        
        # 设置主题
        style.theme_use('clam')
        
        # 配置按钮样式
        self._configure_button_styles(style)
        
        # 配置框架样式
        self._configure_frame_styles(style)
        
        # 配置标签样式
        self._configure_label_styles(style)
        
        # 配置输入框样式
        self._configure_entry_styles(style)
        
        # 配置组合框样式
        self._configure_combobox_styles(style)
        
        # 配置选项卡样式
        self._configure_notebook_styles(style)
        
        # 配置树形视图样式
        self._configure_treeview_styles(style)
        
        return style
    
    def _configure_button_styles(self, style):
        """配置按钮样式"""
        # 主按钮
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
        
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_dark']),
                           ('pressed', self.colors['primary_dark'])])
        
        # 次按钮
        style.configure('Secondary.TButton',
                       background=self.colors['light'],
                       foreground=self.colors['text_primary'],
                       borderwidth=1,
                       focuscolor='none',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
        
        style.map('Secondary.TButton',
                 background=[('active', self.colors['border']),
                           ('pressed', self.colors['border'])])
        
        # 成功按钮
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
        
        # 警告按钮
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground=self.colors['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
        
        # 危险按钮
        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
    
    def _configure_frame_styles(self, style):
        """配置框架样式"""
        # 主框架
        style.configure('Card.TFrame',
                       background=self.colors['bg_card'],
                       borderwidth=1,
                       relief='solid')
        
        # 侧边栏框架
        style.configure('Sidebar.TFrame',
                       background=self.colors['bg_sidebar'])
        
        # 内容框架
        style.configure('Content.TFrame',
                       background=self.colors['bg_primary'])
    
    def _configure_label_styles(self, style):
        """配置标签样式"""
        # 标题标签
        style.configure('Title.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=self.fonts['title'])
        
        # 标题标签
        style.configure('Heading.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=self.fonts['heading'])
        
        # 副标题标签
        style.configure('Subheading.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=self.fonts['subheading'])
        
        # 正文标签
        style.configure('Body.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_primary'],
                       font=self.fonts['body'])
        
        # 次文字标签
        style.configure('Secondary.TLabel',
                       background=self.colors['bg_primary'],
                       foreground=self.colors['text_secondary'],
                       font=self.fonts['body'])
        
        # 白色文字标签
        style.configure('White.TLabel',
                       background=self.colors['bg_sidebar'],
                       foreground=self.colors['text_white'],
                       font=self.fonts['body'])
    
    def _configure_entry_styles(self, style):
        """配置输入框样式"""
        style.configure('Modern.TEntry',
                       fieldbackground=self.colors['white'],
                       borderwidth=1,
                       relief='solid',
                       font=self.fonts['body'],
                       padding=(self.spacing['md'], self.spacing['sm']))
        
        style.map('Modern.TEntry',
                 bordercolor=[('focus', self.colors['border_focus'])])
    
    def _configure_combobox_styles(self, style):
        """配置组合框样式"""
        style.configure('Modern.TCombobox',
                       fieldbackground=self.colors['white'],
                       borderwidth=1,
                       relief='solid',
                       font=self.fonts['body'],
                       padding=(self.spacing['md'], self.spacing['sm']))
        
        style.map('Modern.TCombobox',
                 bordercolor=[('focus', self.colors['border_focus'])])
    
    def _configure_notebook_styles(self, style):
        """配置选项卡样式"""
        style.configure('Modern.TNotebook',
                       background=self.colors['bg_primary'],
                       borderwidth=0)
        
        style.configure('Modern.TNotebook.Tab',
                       background=self.colors['light'],
                       foreground=self.colors['text_primary'],
                       borderwidth=1,
                       relief='solid',
                       font=self.fonts['body'],
                       padding=(self.spacing['lg'], self.spacing['md']))
        
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', self.colors['primary']),
                           ('active', self.colors['primary_light'])],
                 foreground=[('selected', self.colors['white']),
                           ('active', self.colors['white'])])
    
    def _configure_treeview_styles(self, style):
        """配置树形视图样式"""
        style.configure('Modern.Treeview',
                       background=self.colors['white'],
                       foreground=self.colors['text_primary'],
                       fieldbackground=self.colors['white'],
                       borderwidth=1,
                       relief='solid',
                       font=self.fonts['body'])
        
        style.configure('Modern.Treeview.Heading',
                       background=self.colors['light'],
                       foreground=self.colors['text_primary'],
                       borderwidth=1,
                       relief='solid',
                       font=self.fonts['subheading'])
        
        style.map('Modern.Treeview',
                 background=[('selected', self.colors['primary_light'])],
                 foreground=[('selected', self.colors['text_primary'])])
    
    def create_card_frame(self, parent, **kwargs):
        """创建卡片样式的框架"""
        frame = tk.Frame(parent, 
                        bg=self.colors['bg_card'],
                        relief='solid',
                        bd=1,
                        **kwargs)
        return frame
    
    def create_rounded_button(self, parent, text, command=None, style='Primary'):
        """创建圆角按钮"""
        button = tk.Button(parent,
                          text=text,
                          command=command,
                          bg=self.colors['primary'] if style == 'Primary' else self.colors['light'],
                          fg=self.colors['white'] if style == 'Primary' else self.colors['text_primary'],
                          font=self.fonts['body'],
                          relief='flat',
                          bd=0,
                          padx=self.spacing['lg'],
                          pady=self.spacing['md'],
                          cursor='hand2')
        
        # 添加悬停效果
        def on_enter(e):
            if style == 'Primary':
                button.config(bg=self.colors['primary_dark'])
            else:
                button.config(bg=self.colors['border'])
        
        def on_leave(e):
            if style == 'Primary':
                button.config(bg=self.colors['primary'])
            else:
                button.config(bg=self.colors['light'])
        
        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)
        
        return button
    
    def create_modern_label(self, parent, text, style='Body'):
        """创建现代标签"""
        label = tk.Label(parent,
                        text=text,
                        bg=self.colors['bg_primary'],
                        fg=self.colors['text_primary'],
                        font=self.fonts[style.lower()])
        return label
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = tk.Frame(parent,
                               bg=self.colors['light'],
                               relief='sunken',
                               bd=1)
        
        status_label = tk.Label(status_frame,
                               text="就绪",
                               bg=self.colors['light'],
                               fg=self.colors['text_secondary'],
                               font=self.fonts['small'],
                               anchor='w')
        status_label.pack(side='left', padx=self.spacing['md'])
        
        return status_frame, status_label
    
    def create_progress_bar(self, parent):
        """创建进度条"""
        progress_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        
        progress_bar = ttk.Progressbar(progress_frame,
                                     mode='indeterminate',
                                     style='Modern.Horizontal.TProgressbar')
        
        progress_label = tk.Label(progress_frame,
                                 text="处理中...",
                                 bg=self.colors['bg_primary'],
                                 fg=self.colors['text_secondary'],
                                 font=self.fonts['small'])
        
        return progress_frame, progress_bar, progress_label


# 全局主题实例
theme = ModernTheme()
