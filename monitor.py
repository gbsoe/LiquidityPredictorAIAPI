#!/usr/bin/env python3
import os
import sys
import time
import logging
import sqlite3
import psutil
import requests
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitor.log')
    ]
)

logger = logging.getLogger('monitor')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['DATABASE_PATH'] = os.path.join(BASE_DIR, 'database/liquidity_pools.db')
DATABASE_PATH = os.environ['DATABASE_PATH']

# Monitoring parameters
CHECK_INTERVAL = 60  # seconds
SERVICE_TIMEOUT = 5  # seconds
MAX_DB_AGE = 3600    # seconds (1 hour)

# Service endpoints
SERVICES = {
    "backend": "http://localhost:8000/api/health",
    "dashboard": "http://localhost:5000"
}

class SystemMonitor:
    """System monitoring utility for the Liquidity Pool Analysis System."""
    
    def __init__(self):
        """Initialize the system monitor."""
        self.services_status = {}
        self.database_status = {"connected": False, "last_update": None}
        self.processes = []
    
    def check_service(self, name, url):
        """
        Check if a service is running by making an HTTP request.
        
        Args:
            name: Service name
            url: Service URL
        
        Returns:
            Status dictionary
        """
        status = {
            "name": name,
            "url": url,
            "running": False,
            "response_time": None,
            "error": None,
            "last_checked": datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=SERVICE_TIMEOUT)
            elapsed = time.time() - start_time
            
            status["response_time"] = round(elapsed * 1000, 2)  # ms
            status["running"] = response.status_code == 200
            status["status_code"] = response.status_code
            
            if response.status_code != 200:
                status["error"] = f"HTTP {response.status_code}"
            
        except requests.exceptions.ConnectionError:
            status["error"] = "Connection refused"
        except requests.exceptions.Timeout:
            status["error"] = "Request timed out"
        except Exception as e:
            status["error"] = str(e)
        
        self.services_status[name] = status
        return status
    
    def check_database(self):
        """
        Check database connectivity and data freshness.
        
        Returns:
            Status dictionary
        """
        status = {
            "connected": False,
            "tables": [],
            "row_counts": {},
            "last_update": None,
            "error": None,
            "last_checked": datetime.now().isoformat()
        }
        
        try:
            # Try to connect to the database
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            status["connected"] = True
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            status["tables"] = tables
            
            # Count rows in each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                status["row_counts"][table] = count
            
            # Check last update time for pool_metrics
            if "pool_metrics" in tables:
                cursor.execute("SELECT MAX(timestamp) FROM pool_metrics")
                last_update = cursor.fetchone()[0]
                
                if last_update:
                    try:
                        # Parse the timestamp string to a datetime object
                        last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        status["last_update"] = last_update
                        
                        # Check if data is stale
                        age = (datetime.now() - last_update_dt).total_seconds()
                        status["data_age"] = age
                        status["data_stale"] = age > MAX_DB_AGE
                    except Exception as e:
                        logger.warning(f"Error parsing timestamp: {e}")
                        status["error"] = f"Error parsing timestamp: {e}"
            
            conn.close()
            
        except Exception as e:
            status["error"] = str(e)
        
        self.database_status = status
        return status
    
    def find_system_processes(self):
        """
        Find system processes related to our application.
        
        Returns:
            List of process information dictionaries
        """
        processes = []
        
        # Process names to look for
        targets = {
            "node": "Backend Server",
            "python": "Python Process"
        }
        
        # Look for relevant processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                name = proc.info['name']
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Check if this is one of our processes
                if name in targets:
                    # Further filter Python processes
                    if name == "python" and not any(x in cmdline for x in ['app.py', 'data_collector.py', 'price_tracker.py', 'ml_models.py']):
                        continue
                    
                    # Add CPU and memory info
                    cpu = proc.info['cpu_percent']
                    memory = proc.info['memory_percent']
                    
                    # Format create time
                    create_time = datetime.fromtimestamp(proc.info['create_time']).isoformat()
                    
                    # Identify specific component
                    component = targets[name]
                    if "server.js" in cmdline:
                        component = "Node.js Backend"
                    elif "app.py" in cmdline:
                        component = "Streamlit Dashboard"
                    elif "data_collector.py" in cmdline:
                        component = "Data Collector"
                    elif "price_tracker.py" in cmdline:
                        component = "Price Tracker"
                    elif "ml_models.py" in cmdline:
                        component = "ML Models"
                    
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": name,
                        "component": component,
                        "cmdline": cmdline,
                        "cpu_percent": cpu,
                        "memory_percent": memory,
                        "create_time": create_time
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        self.processes = processes
        return processes
    
    def generate_report(self):
        """
        Generate a system health report.
        
        Returns:
            Report dictionary
        """
        # Refresh all status information
        for name, url in SERVICES.items():
            self.check_service(name, url)
        
        self.check_database()
        self.find_system_processes()
        
        # Generate the report
        report = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "services": self.services_status,
            "database": self.database_status,
            "processes": self.processes,
            "overall_health": "unknown"
        }
        
        # Determine overall health
        if all(s.get("running", False) for s in self.services_status.values()) and self.database_status["connected"]:
            if self.database_status.get("data_stale", False):
                report["overall_health"] = "warning"
            else:
                report["overall_health"] = "healthy"
        else:
            report["overall_health"] = "unhealthy"
        
        return report
    
    def print_report(self, report=None):
        """
        Print a formatted system health report.
        
        Args:
            report: Optional report dictionary. If None, a new report is generated.
        """
        if report is None:
            report = self.generate_report()
        
        print("\n" + "=" * 80)
        print(f"SYSTEM HEALTH REPORT - {report['timestamp']}")
        print("=" * 80)
        
        # Overall health
        health = report['overall_health']
        health_color = {
            "healthy": "\033[92m",   # Green
            "warning": "\033[93m",   # Yellow
            "unhealthy": "\033[91m", # Red
            "unknown": "\033[94m"    # Blue
        }.get(health, "\033[0m")
        
        print(f"\nOverall Health: {health_color}{health.upper()}\033[0m")
        
        # System resources
        sys_info = report['system']
        print("\nSystem Resources:")
        print(f"CPU Usage: {sys_info['cpu_percent']}%")
        print(f"Memory Usage: {sys_info['memory_percent']}%")
        print(f"Disk Usage: {sys_info['disk_percent']}%")
        
        # Services
        print("\nServices:")
        for name, status in report['services'].items():
            status_str = "✅ ONLINE" if status.get("running", False) else "❌ OFFLINE"
            status_color = "\033[92m" if status.get("running", False) else "\033[91m"
            response_time = status.get("response_time", "N/A")
            response_str = f" - Response: {response_time}ms" if response_time else ""
            error = status.get("error", "")
            error_str = f" - Error: {error}" if error else ""
            
            print(f"{name.capitalize()}: {status_color}{status_str}\033[0m{response_str}{error_str}")
        
        # Database
        db_status = report['database']
        db_connected = "✅ CONNECTED" if db_status.get("connected", False) else "❌ DISCONNECTED"
        db_color = "\033[92m" if db_status.get("connected", False) else "\033[91m"
        
        print("\nDatabase:")
        print(f"Status: {db_color}{db_connected}\033[0m")
        
        if db_status.get("connected", False):
            # Data freshness
            last_update = db_status.get("last_update", "Never")
            data_age = db_status.get("data_age")
            data_stale = db_status.get("data_stale", True)
            
            age_str = ""
            if data_age is not None:
                # Convert seconds to a readable format
                if data_age < 60:
                    age_str = f"{int(data_age)} seconds ago"
                elif data_age < 3600:
                    age_str = f"{int(data_age/60)} minutes ago"
                else:
                    age_str = f"{int(data_age/3600)} hours ago"
            
            freshness_color = "\033[91m" if data_stale else "\033[92m"
            fresh_status = "STALE" if data_stale else "FRESH"
            print(f"Data Freshness: {freshness_color}{fresh_status}\033[0m - Last Update: {last_update} ({age_str})")
            
            # Table information
            print("\nDatabase Tables:")
            for table, count in db_status.get("row_counts", {}).items():
                print(f"  - {table}: {count} records")
        
        # Processes
        print("\nActive System Processes:")
        for proc in report['processes']:
            print(f"  - {proc['component']} (PID: {proc['pid']}) - CPU: {proc['cpu_percent']}%, Memory: {proc['memory_percent']:.2f}%")
        
        print("\n" + "=" * 80 + "\n")

    def monitor_loop(self):
        """
        Run the monitoring loop continuously.
        """
        try:
            while True:
                report = self.generate_report()
                self.print_report(report)
                
                # Log any issues
                if report['overall_health'] != 'healthy':
                    logger.warning(f"System health: {report['overall_health']}")
                    
                    # Log specific issues
                    for name, status in report['services'].items():
                        if not status.get("running", False):
                            logger.error(f"Service {name} is down: {status.get('error', 'Unknown error')}")
                    
                    if not report['database']['connected']:
                        logger.error(f"Database connection failed: {report['database'].get('error', 'Unknown error')}")
                    elif report['database'].get('data_stale', False):
                        logger.warning("Database data is stale")
                
                # Wait for next check
                time.sleep(CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")

# Main function
def main():
    """Run the system monitor."""
    print("Starting system monitor...")
    monitor = SystemMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Generate a single report and exit
        monitor.print_report()
    else:
        # Run the monitoring loop
        monitor.monitor_loop()

if __name__ == "__main__":
    main()
