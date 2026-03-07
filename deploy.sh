#!/bin/bash
# Deploy to GitHub — รวมคำสั่ง push โค้ดขึ้น GitHub
# Usage: ./deploy.sh [commit message]
# Example: ./deploy.sh "Fix download buttons and email flow"

set -e
MSG="${1:-Deploy: update code}"

echo "📦 Adding changes..."
git add .

echo "📝 Committing: $MSG"
git commit -m "$MSG"

echo "🚀 Pushing to origin main..."
git push origin main

echo "✅ Deploy to GitHub เสร็จแล้ว"
echo "   หมายเหตุ: ถ้าใช้ Render ให้ไปกด Manual Deploy ที่ dashboard"
