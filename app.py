import streamlit as st
import pandas as pd
import numpy as np
import os
import json

from preprocessing import preprocess_pipeline
from recommendation import RestaurantRecommender, MODEL_DIR, METADATA_PATH, TFIDF_MATRIX_PATH
from utils import (
    rating_distribution_chart, cost_distribution_chart,
    cuisine_distribution_chart, top_cities_chart,
    online_order_pie, table_booking_pie,
    top_restaurants_chart, rating_vs_cost_chart,
    popular_restaurants_chart, listed_type_chart,
    rest_type_chart, render_restaurant_card,
)

st.set_page_config(
    page_title="AI Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --neon-green: #00E676;
    --neon-green-dim: #00C853;
    --deep-green: #1B5E20;
    --orange: #FF6D00;
    --blue: #00B0FF;
    --dark-bg: #0a0f1a;
    --dark-card: #111827;
    --text-primary: #f0f0f0;
    --text-secondary: #8892a4;
}

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--dark-bg) !important; color: var(--text-primary) !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,230,118,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(0,176,255,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(255,109,0,0.03) 0%, transparent 50%);
    animation: bgFloat 20s ease-in-out infinite;
    z-index: -1;
    pointer-events: none !important;
}
@keyframes bgFloat {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(2%, -2%) rotate(1deg); }
    66% { transform: translate(-1%, 1%) rotate(-0.5deg); }
}

.particles {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none !important; z-index: -1; overflow: hidden;
}
.particle {
    position: absolute;
    width: 4px; height: 4px;
    background: var(--neon-green);
    border-radius: 50%;
    opacity: 0;
    animation: particleFloat linear infinite;
}
.particle:nth-child(1) { left:10%; animation-duration:18s; animation-delay:0s; }
.particle:nth-child(2) { left:25%; animation-duration:22s; animation-delay:3s; background:var(--blue); }
.particle:nth-child(3) { left:45%; animation-duration:16s; animation-delay:6s; }
.particle:nth-child(4) { left:65%; animation-duration:20s; animation-delay:2s; background:var(--orange); }
.particle:nth-child(5) { left:80%; animation-duration:24s; animation-delay:5s; }
.particle:nth-child(6) { left:35%; animation-duration:19s; animation-delay:8s; background:var(--blue); }
.particle:nth-child(7) { left:55%; animation-duration:21s; animation-delay:1s; }
.particle:nth-child(8) { left:90%; animation-duration:17s; animation-delay:4s; background:var(--neon-green); }
@keyframes particleFloat {
    0% { bottom: -10px; opacity: 0; transform: translateX(0) scale(1); }
    10% { opacity: 0.6; }
    90% { opacity: 0.6; }
    100% { bottom: 110vh; opacity: 0; transform: translateX(80px) scale(0.3); }
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a0f 0%, #111f14 40%, #0a1510 100%) !important;
    border-right: 1px solid rgba(0,230,118,0.15) !important;
    z-index: 9999 !important;
    overflow: visible !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow: visible !important;
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,230,118,0.08) 0%, transparent 70%);
    pointer-events: none !important;
    z-index: 0 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stRadio label span,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label { color: #e0e0e0 !important; }

section[data-testid="stSidebar"] .stRadio > div {
    background: rgba(0,230,118,0.06) !important;
    border: 1px solid rgba(0,230,118,0.12) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
section[data-testid="stSidebar"] .stRadio > div:hover {
    background: rgba(0,230,118,0.12) !important;
    border-color: rgba(0,230,118,0.3) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 0 20px rgba(0,230,118,0.1) !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(26,35,50,0.9)) !important;
    border: 1px solid rgba(0,230,118,0.15) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 40px rgba(0,230,118,0.03) !important;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) rotateX(2deg) !important;
    border-color: rgba(0,230,118,0.4) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 60px rgba(0,230,118,0.08) !important;
}
div[data-testid="stMetric"] label { color: var(--neon-green) !important; font-weight: 600 !important; text-transform: uppercase !important; font-size: 0.75rem !important; letter-spacing: 0.5px !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; font-size: 1.8rem !important; font-family: 'Space Grotesk', sans-serif !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--neon-green), var(--neon-green-dim)) !important;
    color: #0a0f1a !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 36px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 4px 20px rgba(0,230,118,0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s !important;
}
.stButton > button:hover::after { left: 100% !important; }
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(0,230,118,0.5) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(17,24,39,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--neon-green) !important;
    box-shadow: 0 0 0 2px rgba(0,230,118,0.2) !important;
}

.stSlider > div > div > div > div { background: var(--neon-green) !important; }
.stSlider > div > div > div > div > div {
    background: #ffffff !important;
    box-shadow: 0 0 10px rgba(0,230,118,0.5) !important;
}

.stCheckbox > label > div[data-checked="true"] {
    background: var(--neon-green) !important;
    border-color: var(--neon-green) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: rgba(17,24,39,0.5) !important;
    border-radius: 14px !important;
    padding: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,230,118,0.2), rgba(0,230,118,0.1)) !important;
    color: var(--neon-green) !important;
    border-bottom: none !important;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,230,118,0.3), transparent) !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,230,118,0.3); border-radius: 3px; }

.metric-3d {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid rgba(0,230,118,0.15);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
    transform-style: preserve-3d;
    perspective: 1000px;
    animation: metricEntry 0.6s cubic-bezier(0.4,0,0.2,1) forwards;
    opacity: 0; transform: translateY(30px) rotateX(10deg);
}
.metric-3d:nth-child(1) { animation-delay: 0.1s; }
.metric-3d:nth-child(2) { animation-delay: 0.2s; }
.metric-3d:nth-child(3) { animation-delay: 0.3s; }
.metric-3d:nth-child(4) { animation-delay: 0.4s; }
@keyframes metricEntry { to { opacity: 1; transform: translateY(0) rotateX(0); } }
.metric-3d:hover {
    transform: translateY(-8px) rotateX(5deg) rotateY(-3deg) !important;
    border-color: rgba(0,230,118,0.4);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 40px rgba(0,230,118,0.1);
}
.metric-3d-shine {
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,230,118,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.metric-3d-icon { font-size: 2.4rem; margin-bottom: 12px; filter: drop-shadow(0 0 8px rgba(0,230,118,0.4)); animation: iconPulse 3s ease-in-out infinite; }
@keyframes iconPulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
.metric-3d-value { font-size: 2.2rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif; line-height: 1; margin-bottom: 6px; }
.metric-3d-label { color: #8892a4; font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }

.rec-card {
    perspective: 1200px; margin-bottom: 20px;
    animation: cardSlideIn 0.7s cubic-bezier(0.4,0,0.2,1) forwards;
    opacity: 0; transform: translateY(40px);
}
@keyframes cardSlideIn { to { opacity: 1; transform: translateY(0); } }
.rec-card-glow {
    position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(135deg, #00E676, #00B0FF, #FF6D00, #00E676);
    background-size: 400% 400%;
    border-radius: 18px; z-index: -1; opacity: 0;
    animation: glowRotate 4s linear infinite;
    transition: opacity 0.4s ease;
}
.rec-card:hover .rec-card-glow { opacity: 0.6; }
@keyframes glowRotate { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
.rec-card-inner {
    background: linear-gradient(135deg, #111827 0%, #1a2332 50%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 24px;
    position: relative; overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
    transform-style: preserve-3d;
}
.rec-card:hover .rec-card-inner {
    transform: rotateY(-2deg) rotateX(1deg) scale(1.01);
    border-color: rgba(0,230,118,0.3);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 0 60px rgba(0,230,118,0.02);
}
.rec-card-inner::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #00E676, #00B0FF, #FF6D00);
    opacity: 0; transition: opacity 0.4s ease;
}
.rec-card:hover .rec-card-inner::before { opacity: 1; }
.rec-card-inner::after {
    content: '';
    position: absolute; top: -100px; right: -100px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,230,118,0.06) 0%, transparent 70%);
    pointer-events: none;
    transition: all 0.5s ease;
}
.rec-card:hover .rec-card-inner::after { top: -60px; right: -60px; }
.rec-card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.rec-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: linear-gradient(135deg, rgba(0,230,118,0.2), rgba(0,230,118,0.08));
    border: 1px solid rgba(0,230,118,0.3);
    color: #00E676;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    animation: badgeGlow 2s ease-in-out infinite;
}
@keyframes badgeGlow { 0%, 100% { box-shadow: 0 0 5px rgba(0,230,118,0.2); } 50% { box-shadow: 0 0 15px rgba(0,230,118,0.4); } }
.rec-badge-icon { animation: badgeSpin 3s linear infinite; display: inline-block; }
@keyframes badgeSpin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.rec-rating-box { text-align: right; }
.rec-rating-star { color: #FFD600; font-size: 1.3rem; filter: drop-shadow(0 0 4px rgba(255,214,0,0.5)); }
.rec-rating-val { font-size: 1.6rem; font-weight: 800; color: #ffffff; font-family: 'Space Grotesk', sans-serif; margin-left: 4px; }
.rec-rating-count { display: block; color: #8892a4; font-size: 0.75rem; margin-top: 2px; }
.rec-card-title { font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 0 0 8px 0; transition: color 0.3s ease; }
.rec-card:hover .rec-card-title { color: var(--neon-green); }
.rec-card-location { color: #8892a4; font-size: 0.85rem; margin: 4px 0; }
.rec-loc-icon { color: var(--neon-green); margin-right: 4px; }
.rec-card-cuisine { color: #b0b8c4; font-size: 0.85rem; margin: 4px 0; }
.rec-card-type { color: #8892a4; font-size: 0.82rem; margin: 4px 0; }
.rec-card-badges { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 4px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
.badge-online { background: rgba(0,230,118,0.12); color: #00E676; border: 1px solid rgba(0,230,118,0.25); }
.badge-offline { background: rgba(255,23,68,0.1); color: #FF5252; border: 1px solid rgba(255,23,68,0.2); }
.badge-table { background: rgba(0,176,255,0.1); color: #00B0FF; border: 1px solid rgba(0,176,255,0.2); }
.badge-type { background: rgba(255,214,0,0.1); color: #FFD600; border: 1px solid rgba(255,214,0,0.2); }
.rec-card-footer { margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.06); display: flex; align-items: baseline; gap: 6px; }
.rec-price { font-size: 1.4rem; font-weight: 800; color: var(--orange); font-family: 'Space Grotesk', sans-serif; transition: all 0.3s ease; }
.rec-card:hover .rec-price { transform: scale(1.05); display: inline-block; text-shadow: 0 0 20px rgba(255,109,0,0.3); }
.rec-price-label { color: #8892a4; font-size: 0.82rem; }

.hero-section {
    position: relative;
    background: linear-gradient(135deg, #0d2818 0%, #1a4a2e 30%, #0d3320 60%, #0a1f14 100%);
    border-radius: 24px; padding: 50px 40px; text-align: center;
    overflow: hidden; border: 1px solid rgba(0,230,118,0.15); margin-bottom: 36px;
    animation: heroEntry 1s cubic-bezier(0.4,0,0.2,1) forwards;
    opacity: 0; transform: translateY(20px);
}
@keyframes heroEntry { to { opacity: 1; transform: translateY(0); } }
.hero-section::before {
    content: '';
    position: absolute; top: -200px; right: -200px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,230,118,0.1) 0%, transparent 60%);
    animation: heroOrb1 8s ease-in-out infinite;
}
.hero-section::after {
    content: '';
    position: absolute; bottom: -150px; left: -150px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,176,255,0.08) 0%, transparent 60%);
    animation: heroOrb2 10s ease-in-out infinite;
}
@keyframes heroOrb1 { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-40px, 30px); } }
@keyframes heroOrb2 { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(30px, -20px); } }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 3rem; font-weight: 900; color: #ffffff; margin-bottom: 14px; position: relative; z-index: 1; line-height: 1.1; }
.hero-title span { background: linear-gradient(135deg, #00E676, #00B0FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-subtitle { color: #b0b8c4; font-size: 1.1rem; max-width: 700px; margin: 0 auto; position: relative; z-index: 1; line-height: 1.6; }
.hero-particles { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none !important; z-index: -1; }
.hero-dot { position: absolute; width: 3px; height: 3px; background: rgba(0,230,118,0.4); border-radius: 50%; animation: heroDotFloat 6s ease-in-out infinite; }
.hero-dot:nth-child(1) { top:20%; left:15%; animation-delay:0s; }
.hero-dot:nth-child(2) { top:60%; left:80%; animation-delay:1.5s; background:rgba(0,176,255,0.4); }
.hero-dot:nth-child(3) { top:80%; left:30%; animation-delay:3s; }
.hero-dot:nth-child(4) { top:30%; left:70%; animation-delay:0.8s; background:rgba(255,109,0,0.4); }
.hero-dot:nth-child(5) { top:45%; left:50%; animation-delay:2.2s; }
@keyframes heroDotFloat { 0%, 100% { transform: translateY(0) scale(1); opacity: 0.4; } 50% { transform: translateY(-15px) scale(1.5); opacity: 0.8; } }

.workflow-step {
    background: linear-gradient(135deg, #111827, #1a2332);
    border: 1px solid rgba(0,230,118,0.1);
    border-radius: 18px; padding: 24px 16px; text-align: center;
    position: relative; overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
    transform-style: preserve-3d;
    animation: stepEntry 0.6s cubic-bezier(0.4,0,0.2,1) forwards;
    opacity: 0; transform: translateY(20px) rotateX(8deg);
}
.workflow-step:nth-child(1) { animation-delay: 0.15s; }
.workflow-step:nth-child(2) { animation-delay: 0.25s; }
.workflow-step:nth-child(3) { animation-delay: 0.35s; }
.workflow-step:nth-child(4) { animation-delay: 0.45s; }
.workflow-step:nth-child(5) { animation-delay: 0.55s; }
@keyframes stepEntry { to { opacity: 1; transform: translateY(0) rotateX(0); } }
.workflow-step:hover {
    transform: translateY(-10px) rotateY(5deg) !important;
    border-color: rgba(0,230,118,0.4);
    box-shadow: 0 16px 50px rgba(0,0,0,0.4), 0 0 30px rgba(0,230,118,0.08);
}
.workflow-step::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00E676, #00B0FF);
    transform: scaleX(0); transition: transform 0.4s ease;
}
.workflow-step:hover::after { transform: scaleX(1); }
.workflow-icon { font-size: 2.2rem; margin-bottom: 12px; display: inline-block; animation: iconFloat 3s ease-in-out infinite; filter: drop-shadow(0 0 10px rgba(0,230,118,0.3)); }
@keyframes iconFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
.workflow-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #ffffff; font-size: 0.95rem; margin-bottom: 6px; }
.workflow-desc { color: #8892a4; font-size: 0.78rem; line-height: 1.4; }

.about-card {
    background: linear-gradient(135deg, #111827, #1a2332);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px; padding: 28px; margin-bottom: 20px;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
}
.about-card:hover { border-color: rgba(0,230,118,0.25); box-shadow: 0 8px 40px rgba(0,0,0,0.3); transform: translateY(-3px); }
.about-card h3 { color: var(--neon-green) !important; font-family: 'Space Grotesk', sans-serif; }
.about-card p, .about-card li { color: #b0b8c4; line-height: 1.7; }

.tech-pill {
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(0,230,118,0.02));
    border: 1px solid rgba(0,230,118,0.15);
    border-radius: 14px; padding: 18px 14px; text-align: center;
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
    animation: pillEntry 0.5s cubic-bezier(0.4,0,0.2,1) forwards;
    opacity: 0; transform: scale(0.8);
}
.tech-pill:nth-child(1) { animation-delay: 0.05s; }
.tech-pill:nth-child(2) { animation-delay: 0.1s; }
.tech-pill:nth-child(3) { animation-delay: 0.15s; }
.tech-pill:nth-child(4) { animation-delay: 0.2s; }
.tech-pill:nth-child(5) { animation-delay: 0.25s; }
.tech-pill:nth-child(6) { animation-delay: 0.3s; }
.tech-pill:nth-child(7) { animation-delay: 0.35s; }
.tech-pill:nth-child(8) { animation-delay: 0.4s; }
@keyframes pillEntry { to { opacity: 1; transform: scale(1); } }
.tech-pill:hover { transform: translateY(-5px) scale(1.05) !important; border-color: var(--neon-green); box-shadow: 0 10px 30px rgba(0,230,118,0.15); }
.tech-pill-icon { font-size: 1.8rem; margin-bottom: 8px; display: block; }
.tech-pill-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #ffffff; font-size: 0.85rem; display: block; margin-bottom: 3px; }
.tech-pill-desc { color: #8892a4; font-size: 0.72rem; }

.pref-panel {
    background: linear-gradient(135deg, #111827, #1a2332);
    border: 1px solid rgba(0,230,118,0.12);
    border-radius: 20px; padding: 28px;
    position: relative; overflow: hidden;
}
.pref-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00E676, #00B0FF, #FF6D00);
}

.results-header {
    background: linear-gradient(135deg, rgba(0,230,118,0.12), rgba(0,176,255,0.08));
    border: 1px solid rgba(0,230,118,0.2);
    color: #ffffff; padding: 16px 24px; border-radius: 14px; margin-bottom: 24px;
}
.results-header h3 { color: #ffffff !important; margin: 0 !important; }
.results-header p { color: rgba(255,255,255,0.7) !important; margin: 4px 0 0 0 !important; }

.empty-state { text-align: center; padding: 80px 20px; color: #8892a4; }
.empty-state-icon { font-size: 5rem; margin-bottom: 20px; animation: emptyPulse 3s ease-in-out infinite; display: inline-block; }
@keyframes emptyPulse { 0%, 100% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.1) rotate(5deg); } }
.empty-state h3 { color: var(--neon-green); font-family: 'Space Grotesk', sans-serif; }
.empty-state p { color: #8892a4; max-width: 400px; margin: 10px auto 0; line-height: 1.6; }

.pulse-dot { width: 8px; height: 8px; background: var(--neon-green); border-radius: 50%; display: inline-block; margin-right: 8px; animation: pulseDot 2s ease-in-out infinite; }
@keyframes pulseDot { 0%, 100% { box-shadow: 0 0 0 0 rgba(0,230,118,0.4); } 50% { box-shadow: 0 0 0 8px rgba(0,230,118,0); } }
</style>

<div class="particles">
    <div class="particle"></div><div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div><div class="particle"></div>
    <div class="particle"></div><div class="particle"></div>
</div>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")


def ensure_model_exists():
    """Make sure the trained model files exist before loading."""
    if os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(METADATA_PATH):
        return True

    import subprocess
    import sys

    train_script = os.path.join(BASE_DIR, "train_model.py")

    if os.path.exists(train_script):
        try:
            subprocess.run(
                [sys.executable, train_script],
                cwd=BASE_DIR,
                check=True
            )
        except Exception as e:
            st.warning(f"Model training failed: {e}")
            return False

    return (
        os.path.exists(TFIDF_MATRIX_PATH)
        and os.path.exists(METADATA_PATH)
    )


@st.cache_resource
def load_recommender():
    if ensure_model_exists():
        try:
            rec = RestaurantRecommender()
            return rec, rec.metadata
        except Exception as e:
            st.warning(f"Could not load saved model: {e}")

    df, stats = preprocess_pipeline(DATASET_PATH)
    rec = RestaurantRecommender(df)
    return rec, stats


recommender, stats = load_recommender()

recommender, stats = load_recommender()

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px;position:relative;">
        <div style="font-size:3.2rem;margin-bottom:10px;animation:iconFloat 3s ease-in-out infinite;filter:drop-shadow(0 0 12px rgba(0,230,118,0.5));">🍽️</div>
        <h1 style="color:#ffffff;font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;margin:0;">AI Restaurant</h1>
        <p style="color:rgba(0,230,118,0.7);font-size:0.75rem;margin:4px 0 0;font-weight:500;letter-spacing:2px;text-transform:uppercase;">Recommendation System</p>
        <div style="width:40px;height:2px;background:linear-gradient(90deg,#00E676,#00B0FF);margin:12px auto 0;border-radius:1px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigation", ["🏠 Home", "📊 Data Analysis", "🎯 Recommendation", "ℹ️ About"], label_visibility="collapsed")

# ═══════════════════════ HOME ═══════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-particles">
            <div class="hero-dot"></div><div class="hero-dot"></div>
            <div class="hero-dot"></div><div class="hero-dot"></div><div class="hero-dot"></div>
        </div>
        <div class="hero-title">🍽️ AI Restaurant<br><span>Recommendation System</span></div>
        <div class="hero-subtitle">
            Discover the best restaurants tailored to your taste.<br>
            Powered by Machine Learning & NLP for personalized recommendations from <strong>51,000+ restaurants</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🏪 Restaurants", f"{stats['total_restaurants']:,}", "Bangalore Zomato data")
    with c2: st.metric("🏙️ Areas", f"{stats['total_cities']}", "Neighborhoods covered")
    with c3: st.metric("🍽️ Cuisines", f"{stats['total_cuisines']}", "Unique cuisine types")
    with c4: st.metric("⭐ Avg Rating", f"{stats['avg_rating']}", "Across all restaurants")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pulse-dot"></div><span style="color:#00E676;font-weight:700;font-family:Space Grotesk,sans-serif;">HOW IT WORKS</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    wf = st.columns(5)
    steps = [
        ("📊", "Zomato Data", "51K+ restaurants with 17 features"),
        ("🧹", "Cleaning", "Rate parsing, null handling, dedup"),
        ("🔧", "Feature Eng.", "TF-IDF on cuisines, dishes, type"),
        ("🤖", "ML Model", "Cosine Similarity + composite score"),
        ("🎯", "Results", "Top 5 personalized recommendations"),
    ]
    for col, (icon, title, desc) in zip(wf, steps):
        with col:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="workflow-icon">{icon}</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="pulse-dot"></div><span style="color:#00E676;font-weight:700;font-family:Space Grotesk,sans-serif;">DATASET OVERVIEW</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        recommender.df.head(10)[["name","location","cuisines","rate","votes","approx_cost(for two people)","listed_in(type)"]].style.format(
            {"rate":"{:.1f}","votes":"{:,.0f}","approx_cost(for two people)":"₹{:,.0f}"}
        ),
        use_container_width=True,
    )


# ═══════════════════════ DATA ANALYSIS ═══════════════════════
elif page == "📊 Data Analysis":
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="color:#ffffff;font-family:'Space Grotesk',sans-serif;margin:0;">
            <span style="color:#00E676;">📊</span> Data Analysis
        </h1>
        <p style="color:#8892a4;margin:6px 0 0;">Explore {0:,} restaurants from Bangalore with interactive charts</p>
    </div>
    """.format(len(recommender.df)), unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏆 Top Restaurants", "🍽️ Cuisines & Areas", "💰 Cost & Rating", "📱 Features", "📋 Types"]
    )

    with tab1:
        st.plotly_chart(top_restaurants_chart(recommender.df), use_container_width=True)
        st.plotly_chart(popular_restaurants_chart(recommender.df), use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(cuisine_distribution_chart(recommender.df), use_container_width=True)
        with c2: st.plotly_chart(top_cities_chart(recommender.df), use_container_width=True)
    with tab3:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(rating_distribution_chart(recommender.df), use_container_width=True)
        with c2: st.plotly_chart(cost_distribution_chart(recommender.df), use_container_width=True)
        st.plotly_chart(rating_vs_cost_chart(recommender.df), use_container_width=True)
    with tab4:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(online_order_pie(recommender.df), use_container_width=True)
        with c2: st.plotly_chart(table_booking_pie(recommender.df), use_container_width=True)
        st.markdown("### 🔍 Top Restaurants by Votes")
        top_voted = recommender.df.nlargest(15, "votes")[["name","rate","votes","cuisines","location","approx_cost(for two people)"]].reset_index(drop=True)
        top_voted.index += 1
        st.dataframe(
            top_voted.style.format({"rate":"{:.1f}","votes":"{:,.0f}","approx_cost(for two people)":"₹{:,.0f}"}),
            use_container_width=True,
        )
    with tab5:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(listed_type_chart(recommender.df), use_container_width=True)
        with c2: st.plotly_chart(rest_type_chart(recommender.df), use_container_width=True)


# ═══════════════════════ RECOMMENDATION ═══════════════════════
elif page == "🎯 Recommendation":
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="color:#ffffff;font-family:'Space Grotesk',sans-serif;margin:0;">
            <span style="color:#00E676;">🎯</span> Restaurant Recommendation
        </h1>
        <p style="color:#8892a4;margin:6px 0 0;">Find your perfect restaurant from {0:,} options powered by AI</p>
    </div>
    """.format(len(recommender.df)), unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.markdown("""<div class="pref-panel"><h3 style="color:#00E676;font-family:Space Grotesk,sans-serif;margin-top:0;">🎛️ Your Preferences</h3>""", unsafe_allow_html=True)

        city = st.selectbox("🏙️ Area in Bangalore", ["All"] + stats["cities"], index=0)
        cuisine = st.selectbox("🍽️ Cuisine", ["All"] + stats["cuisines"], index=0)
        budget = st.slider("💰 Budget (For Two)", 100, 6000, 2000, 100, format="₹%d")
        min_rating = st.slider("⭐ Minimum Rating", 3.0, 5.0, 3.5, 0.1)
        c1, c2 = st.columns(2)
        with c1: online_order = st.checkbox("📱 Online Order")
        with c2: table_booking = st.checkbox("🪑 Table Booking")
        recommend_btn = st.button("🔍 Find Restaurants", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        if recommend_btn:
            with st.spinner("🤖 Finding the best restaurants for you..."):
                import time; time.sleep(0.5)
                results = recommender.get_recommendations(
                    city=city, cuisine=cuisine, budget=budget,
                    min_rating=min_rating, online_order=online_order,
                    table_booking=table_booking, top_n=5,
                )
            if results.empty:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">😔</div>
                    <h3>No Restaurants Found</h3>
                    <p>Try adjusting your filters — broaden the budget, lower the rating, or change the cuisine.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="results-header">
                    <h3>🏆 Top {len(results)} Recommendations</h3>
                    <p>Based on your preferences</p>
                </div>
                """, unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    st.markdown(render_restaurant_card(row, i), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🍽️</div>
                <h3>Set Your Preferences</h3>
                <p>Select your area, cuisine, budget, and other preferences, then click <strong>Find Restaurants</strong> to discover your perfect meal.</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════ ABOUT ═══════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div style="margin-bottom:28px;">
        <h1 style="color:#ffffff;font-family:'Space Grotesk',sans-serif;margin:0;">
            <span style="color:#00E676;">ℹ️</span> About This Project
        </h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h3>📋 Problem Statement</h3>
        <p>With over 51,000 restaurants in Bangalore alone, diners are overwhelmed with choices. This system helps users
        find the perfect restaurant based on their preferences using AI-powered recommendations.
        It leverages content-based filtering with TF-IDF vectorization and cosine similarity
        to suggest restaurants matching the user's desired cuisine, budget, location, and other criteria.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <h3>🎯 Objectives</h3>
        <ul>
            <li>Build a restaurant recommendation system using content-based filtering</li>
            <li>Use TF-IDF vectorization on cuisines, dishes liked, restaurant type, and location</li>
            <li>Apply cosine similarity + composite scoring for restaurant matching</li>
            <li>Create an interactive dark-theme UI using Streamlit with 3D animations</li>
            <li>Enable filtering by area, cuisine, budget, rating, online order, and table booking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="about-card"><h3>🛠️ Technologies Used</h3></div>""", unsafe_allow_html=True)

    techs = [
        ("🐍","Python","Core language"), ("🐼","Pandas","Data manipulation"),
        ("🔢","NumPy","Numerical computing"), ("🤖","Scikit-learn","ML algorithms"),
        ("📊","TF-IDF","Text vectorization"), ("📐","Cosine Similarity","Similarity matching"),
        ("🌐","Streamlit","Web application"), ("📈","Plotly","Interactive charts"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(techs):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="tech-pill">
                <span class="tech-pill-icon">{icon}</span>
                <span class="tech-pill-name">{name}</span>
                <span class="tech-pill-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <h3>🔄 Machine Learning Workflow</h3>
        <ol>
            <li><strong>Data Collection:</strong> Zomato Bangalore dataset with 51,717 restaurants and 17 features</li>
            <li><strong>Data Cleaning:</strong> Parse ratings (remove "/5"), convert costs, handle 7,775 missing ratings, remove duplicates</li>
            <li><strong>Feature Engineering:</strong> Create combined text features from cuisines, dishes liked, restaurant type, listing type, and location</li>
            <li><strong>TF-IDF Vectorization:</strong> Convert text features into numerical vectors using bigram TF-IDF (5,000 features)</li>
            <li><strong>Cosine Similarity:</strong> Compute pairwise similarity scores between all restaurants</li>
            <li><strong>Composite Scoring:</strong> Combine similarity score (25%) + rating (35%) + cost efficiency (20%) + popularity (20%)</li>
            <li><strong>Recommendation:</strong> Filter by user preferences, rank by composite score, return top 5</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,230,118,0.1),rgba(0,176,255,0.06));border:1px solid rgba(0,230,118,0.2);border-radius:18px;padding:28px;text-align:center;">
        <h3 style="color:#00E676;font-family:Space Grotesk,sans-serif;">✨ Key Features</h3>
        <p style="color:#b0b8c4;line-height:2;">
            Content-Based Filtering &bull; TF-IDF Bigrams &bull; Cosine Similarity &bull; Composite Scoring<br>
            Interactive Filters &bull; Dark Theme UI &bull; 3D Animations &bull; Real-time Recommendations<br>
            Plotly Visualizations &bull; Dish Liked Analysis &bull; 51K+ Restaurant Database
        </p>
    </div>
    """, unsafe_allow_html=True)
