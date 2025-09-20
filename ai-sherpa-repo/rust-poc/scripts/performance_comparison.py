#!/usr/bin/env python3
"""
Performance Comparison Script for Silva PoC
Compares the new Rust implementation against the original Python multi-container setup

Usage:
    python performance_comparison.py --old-url http://localhost:8001 --new-url http://localhost:8000
"""

import asyncio
import aiohttp
import time
import statistics
import json
import argparse
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    response_times: List[float]
    memory_usage: float
    cpu_usage: float
    error_rate: float
    throughput: float
    cold_start_time: float
    image_size_mb: float

class PerformanceTester:
    """Performance testing utility"""
    
    def __init__(self, base_url: str, name: str):
        self.base_url = base_url
        self.name = name
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status and system metrics"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    async def measure_cold_start(self) -> float:
        """Measure cold start time by checking when service becomes available"""
        print(f"📊 Measuring cold start time for {self.name}...")
        
        start_time = time.time()
        max_attempts = 60  # 60 seconds timeout
        
        for attempt in range(max_attempts):
            try:
                health = await self.health_check()
                if health.get("status") == "healthy":
                    cold_start_time = time.time() - start_time
                    print(f"✅ {self.name} cold start: {cold_start_time:.3f}s")
                    return cold_start_time
            except:
                pass
            
            await asyncio.sleep(1)
        
        print(f"❌ {self.name} failed to start within {max_attempts}s")
        return float('inf')
    
    async def single_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single request and measure response time"""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": data
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e)
            }
    
    async def load_test(self, num_requests: int = 100, concurrency: int = 10) -> List[Dict[str, Any]]:
        """Run load test with specified parameters"""
        print(f"🚀 Running load test for {self.name}: {num_requests} requests, {concurrency} concurrent")
        
        # Test payloads for different endpoints
        test_payloads = [
            {
                "endpoint": "/route",
                "payload": {"message": "Write a Python function to sort a list", "language": "en"}
            },
            {
                "endpoint": "/route", 
                "payload": {"message": "பைத்தான் நிரலை எழுதுங்கள்", "language": "ta"}
            },
            {
                "endpoint": "/agent",
                "payload": {"agent": "code", "message": "Create hello world", "language": "en"}
            }
        ]
        
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request(i: int) -> Dict[str, Any]:
            async with semaphore:
                test_case = test_payloads[i % len(test_payloads)]
                result = await self.single_request(test_case["endpoint"], test_case["payload"])
                result["request_id"] = i
                result["endpoint"] = test_case["endpoint"]
                return result
        
        # Execute all requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        
        print(f"✅ {self.name} load test completed: {len(valid_results)}/{num_requests} successful")
        
        return valid_results, total_time
    
    async def get_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        print(f"📊 Collecting metrics for {self.name}...")
        
        # Get system health
        health = await self.health_check()
        
        # Run load test
        load_results, total_time = await self.load_test(num_requests=50, concurrency=5)
        
        # Calculate metrics
        successful_requests = [r for r in load_results if r.get("success", False)]
        response_times = [r["response_time"] for r in successful_requests]
        
        error_rate = (len(load_results) - len(successful_requests)) / len(load_results) if load_results else 1.0
        throughput = len(successful_requests) / total_time if total_time > 0 else 0
        
        return PerformanceMetrics(
            response_times=response_times,
            memory_usage=health.get("memory_usage", 0),
            cpu_usage=health.get("cpu_usage", 0),
            error_rate=error_rate,
            throughput=throughput,
            cold_start_time=0,  # Measured separately
            image_size_mb=0     # Measured separately
        )

def calculate_improvement(old_value: float, new_value: float) -> float:
    """Calculate percentage improvement"""
    if old_value == 0:
        return 0
    return ((old_value - new_value) / old_value) * 100

def format_improvement(improvement: float) -> str:
    """Format improvement percentage with color coding"""
    if improvement > 0:
        return f"\033[92m-{improvement:.1f}%\033[0m"  # Green for improvement
    elif improvement < 0:
        return f"\033[91m+{abs(improvement):.1f}%\033[0m"  # Red for regression
    else:
        return "0.0%"

async def compare_systems(old_url: str, new_url: str) -> None:
    """Compare performance between old and new systems"""
    print("🔬 Silva PoC Performance Comparison")
    print("=" * 50)
    
    # Test both systems
    async with PerformanceTester(old_url, "Python Multi-Container") as old_tester:
        async with PerformanceTester(new_url, "Rust Silva PoC") as new_tester:
            
            # Check if both systems are available
            old_health = await old_tester.health_check()
            new_health = await new_tester.health_check()
            
            if old_health.get("status") != "healthy":
                print(f"❌ Old system not available: {old_health.get('error', 'Unknown error')}")
                return
            
            if new_health.get("status") != "healthy":
                print(f"❌ New system not available: {new_health.get('error', 'Unknown error')}")
                return
            
            print("✅ Both systems are healthy, starting comparison...\n")
            
            # Collect metrics
            old_metrics = await old_tester.get_metrics()
            new_metrics = await new_tester.get_metrics()
            
            # Calculate statistics
            old_avg_response = statistics.mean(old_metrics.response_times) * 1000  # Convert to ms
            new_avg_response = statistics.mean(new_metrics.response_times) * 1000
            
            old_p95_response = statistics.quantiles(old_metrics.response_times, n=20)[18] * 1000
            new_p95_response = statistics.quantiles(new_metrics.response_times, n=20)[18] * 1000
            
            # Print comparison table
            print("📊 Performance Comparison Results")
            print("=" * 80)
            print(f"{'Metric':<25} {'Old System':<20} {'New System':<20} {'Improvement':<15}")
            print("-" * 80)
            
            # Response time metrics
            response_improvement = calculate_improvement(old_avg_response, new_avg_response)
            print(f"{'Avg Response Time':<25} {old_avg_response:.1f}ms{'':<12} {new_avg_response:.1f}ms{'':<12} {format_improvement(response_improvement):<15}")
            
            p95_improvement = calculate_improvement(old_p95_response, new_p95_response)
            print(f"{'P95 Response Time':<25} {old_p95_response:.1f}ms{'':<12} {new_p95_response:.1f}ms{'':<12} {format_improvement(p95_improvement):<15}")
            
            # Memory usage
            memory_improvement = calculate_improvement(old_metrics.memory_usage, new_metrics.memory_usage)
            print(f"{'Memory Usage':<25} {old_metrics.memory_usage:.1f}MB{'':<12} {new_metrics.memory_usage:.1f}MB{'':<12} {format_improvement(memory_improvement):<15}")
            
            # Throughput
            throughput_improvement = calculate_improvement(1/old_metrics.throughput if old_metrics.throughput > 0 else float('inf'), 
                                                         1/new_metrics.throughput if new_metrics.throughput > 0 else float('inf'))
            print(f"{'Throughput':<25} {old_metrics.throughput:.1f} req/s{'':<8} {new_metrics.throughput:.1f} req/s{'':<8} {format_improvement(-throughput_improvement):<15}")
            
            # Error rate
            error_improvement = calculate_improvement(old_metrics.error_rate * 100, new_metrics.error_rate * 100)
            print(f"{'Error Rate':<25} {old_metrics.error_rate*100:.2f}%{'':<14} {new_metrics.error_rate*100:.2f}%{'':<14} {format_improvement(error_improvement):<15}")
            
            print("\n" + "=" * 80)
            
            # Summary
            print("\n📈 Summary")
            print(f"• Response time improved by {response_improvement:.1f}%")
            print(f"• Memory usage reduced by {memory_improvement:.1f}%")
            print(f"• Throughput {'increased' if throughput_improvement < 0 else 'decreased'} by {abs(throughput_improvement):.1f}%")
            print(f"• Error rate {'reduced' if error_improvement > 0 else 'increased'} by {abs(error_improvement):.1f}%")
            
            # Theoretical improvements (from design goals)
            print("\n🎯 Design Goals vs Reality")
            print(f"• RAM usage target: -80% (Actual: {memory_improvement:.1f}%)")
            print(f"• Response time target: -60% (Actual: {response_improvement:.1f}%)")
            
            # Recommendations
            print("\n💡 Recommendations")
            if memory_improvement < 70:
                print("• Memory usage could be optimized further")
            if response_improvement < 50:
                print("• Response time optimization needed")
            if new_metrics.error_rate > 0.01:
                print("• Error rate is above 1% target")
            
            if (memory_improvement > 70 and response_improvement > 50 and 
                new_metrics.error_rate < 0.01):
                print("✅ All performance targets met! Ready for production.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compare Silva PoC performance")
    parser.add_argument("--old-url", default="http://localhost:8001", 
                       help="URL of the old Python system")
    parser.add_argument("--new-url", default="http://localhost:8000", 
                       help="URL of the new Rust system")
    parser.add_argument("--requests", type=int, default=100,
                       help="Number of requests for load testing")
    parser.add_argument("--concurrency", type=int, default=10,
                       help="Concurrent requests")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(compare_systems(args.old_url, args.new_url))
    except KeyboardInterrupt:
        print("\n❌ Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage:
# python performance_comparison.py --old-url http://localhost:8001 --new-url http://localhost:8000
# 
# Expected output:
# 📊 Performance Comparison Results
# ================================================================================
# Metric                    Old System           New System           Improvement    
# --------------------------------------------------------------------------------
# Avg Response Time         280.0ms              90.0ms               -67.9%
# P95 Response Time         450.0ms              150.0ms              -66.7%
# Memory Usage              600.0MB              38.0MB               -93.7%
# Throughput                15.2 req/s           45.8 req/s           +201.3%
# Error Rate                2.50%                0.20%                -92.0%