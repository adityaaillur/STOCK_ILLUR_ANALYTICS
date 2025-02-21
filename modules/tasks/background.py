import asyncio
from typing import Callable, Optional
from loguru import logger
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class BackgroundTaskManager:
    """Manages background tasks for data updates"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.tasks = {}
        
    def add_task(self, 
                task_func: Callable,
                name: str,
                interval: timedelta,
                args: Optional[tuple] = None,
                kwargs: Optional[dict] = None):
        """Add a periodic background task"""
        try:
            job = self.scheduler.add_job(
                task_func,
                'interval',
                seconds=interval.total_seconds(),
                args=args,
                kwargs=kwargs,
                id=name
            )
            self.tasks[name] = job
            logger.info(f"Added background task: {name} (interval: {interval})")
        except Exception as e:
            logger.error(f"Error adding background task {name}: {e}")
            
    def start(self):
        """Start the background task scheduler"""
        try:
            self.scheduler.start()
            logger.info("Background task scheduler started")
        except Exception as e:
            logger.error(f"Error starting background scheduler: {e}")
            
    def stop(self):
        """Stop the background task scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Background task scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping background scheduler: {e}")
            
    def get_task_status(self, name: str) -> Optional[dict]:
        """Get status of a specific task"""
        if name in self.tasks:
            job = self.tasks[name]
            return {
                'name': name,
                'next_run': job.next_run_time,
                'last_run': job.last_run_time,
                'runs': job.run_count
            }
        return None 