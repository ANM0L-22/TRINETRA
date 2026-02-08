#!/usr/bin/env python
"""
Simple Flask Dashboard for Vehicle Detection System
Provides web interface for viewing detection results
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database_manager import DatabaseManager

app = Flask(__name__)
CORS(app)

db = DatabaseManager()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Web Routes ====================

@app.route('/')
def index():
    """Dashboard home page"""
    return render_template('dashboard.html')


@app.route('/api/dashboard-stats')
def dashboard_stats():
    """Get dashboard statistics"""
    try:
        stats = db.get_statistics()
        
        return jsonify({
            "status": "success",
            "statistics": stats
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/recent-tracks')
def recent_tracks():
    """Get recent vehicle tracks"""
    try:
        limit = request.args.get('limit', 20, type=int)
        tracks = db.get_all_tracks(limit=limit)
        
        # Enhance with plate info
        for track in tracks:
            plates = db.get_track_plates(track['track_id'])
            track['plates'] = [p['detected_text'] for p in plates if p['detected_text']]
        
        return jsonify({
            "status": "success",
            "tracks": tracks
        })
    except Exception as e:
        logger.error(f"Error getting tracks: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/track/<int:track_id>/details')
def track_details(track_id):
    """Get detailed information about a track"""
    try:
        track = db.get_vehicle_track(track_id)
        
        if not track:
            return jsonify({"status": "error", "message": "Track not found"}), 404
        
        detections = db.get_track_detections(track_id)
        plates = db.get_track_plates(track_id)
        
        return jsonify({
            "status": "success",
            "track": track,
            "detections_count": len(detections),
            "plate_recognitions": plates
        })
    except Exception as e:
        logger.error(f"Error getting track details: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/search/plates')
def search_plates():
    """Search vehicles by license plate"""
    try:
        plate = request.args.get('q', '', type=str)
        
        if len(plate) < 2:
            return jsonify({
                "status": "error",
                "message": "Search query too short"
            }), 400
        
        tracks = db.search_by_plate(plate)
        
        return jsonify({
            "status": "success",
            "query": plate,
            "results": tracks,
            "count": len(tracks)
        })
    except Exception as e:
        logger.error(f"Error searching plates: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/vehicles/by-class')
def vehicles_by_class():
    """Get vehicle count by class"""
    try:
        stats = db.get_statistics()
        
        if 'tracks_by_class' in stats:
            data = [
                {
                    "class": class_name,
                    "count": count
                }
                for class_name, count in stats['tracks_by_class'].items()
            ]
            
            return jsonify({
                "status": "success",
                "data": data
            })
        
        return jsonify({
            "status": "success",
            "data": []
        })
    except Exception as e:
        logger.error(f"Error getting vehicle stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({"status": "healthy", "service": "Vehicle Detection Dashboard"})


if __name__ == '__main__':
    logger.info("Starting Vehicle Detection Dashboard")
    logger.info("Access at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
