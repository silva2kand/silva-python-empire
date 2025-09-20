from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import subprocess
import json
import os
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
# Mock GPT4All service for development
class MockGPT4AllService:
    def __init__(self):
        print("GPT4All not available. Using mock responses for development")
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        return "Mock analysis response for development purposes"

app = FastAPI(title="AI Sherpa - History Analysis Service", version="1.0.0")
gpt_service = MockGPT4AllService()

# Request/Response Models
class HistoryAnalysisRequest(BaseModel):
    repository_path: str
    file_path: Optional[str] = None
    author: Optional[str] = None
    since_date: Optional[str] = None
    until_date: Optional[str] = None
    max_commits: Optional[int] = 100
    include_diffs: bool = False
    analyze_patterns: bool = True

class CommitInfo(BaseModel):
    hash: str
    author: str
    email: str
    date: str
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    diff: Optional[str] = None
    bug_indicators: List[str] = []
    risk_score: float = 0.0

class BugPrediction(BaseModel):
    file_path: str
    risk_score: float
    confidence: float
    factors: List[str]
    recent_changes: int
    bug_history: int
    complexity_indicators: List[str]
    recommendations: List[str]

class HistoryAnalysisResult(BaseModel):
    repository_info: Dict[str, Any]
    commits: List[CommitInfo]
    bug_predictions: List[BugPrediction]
    patterns: Dict[str, Any]
    hotspots: List[Dict[str, Any]]
    author_statistics: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    recommendations: List[str]
    confidence: float

class AuthorAnalysisRequest(BaseModel):
    repository_path: str
    author: Optional[str] = None
    time_period: Optional[str] = "6months"

class AuthorAnalysisResult(BaseModel):
    author_stats: Dict[str, Any]
    contribution_patterns: Dict[str, Any]
    collaboration_network: Dict[str, Any]
    expertise_areas: List[str]
    productivity_trends: Dict[str, Any]
    recommendations: List[str]

@app.get("/")
async def root():
    return {"message": "AI Sherpa History Analysis Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "history-analysis"}

@app.post("/analyze", response_model=HistoryAnalysisResult)
async def analyze_history(request: HistoryAnalysisRequest):
    """Analyze Git history for patterns, bug prediction, and insights."""
    try:
        if not os.path.exists(request.repository_path):
            raise HTTPException(status_code=400, detail="Repository path does not exist")
        
        if not os.path.exists(os.path.join(request.repository_path, ".git")):
            raise HTTPException(status_code=400, detail="Not a Git repository")
        
        # Get repository information
        repo_info = get_repository_info(request.repository_path)
        
        # Get commit history
        commits = get_commit_history(request)
        
        # Analyze commits for bug indicators
        analyzed_commits = analyze_commits_for_bugs(commits)
        
        # Generate bug predictions
        bug_predictions = generate_bug_predictions(request.repository_path, analyzed_commits)
        
        # Analyze patterns
        patterns = analyze_commit_patterns(analyzed_commits) if request.analyze_patterns else {}
        
        # Identify hotspots
        hotspots = identify_hotspots(analyzed_commits)
        
        # Generate author statistics
        author_stats = generate_author_statistics(analyzed_commits)
        
        # Timeline analysis
        timeline = analyze_timeline(analyzed_commits)
        
        # Generate recommendations
        recommendations = generate_history_recommendations(analyzed_commits, bug_predictions, patterns)
        
        # Calculate overall confidence
        confidence = calculate_analysis_confidence(len(analyzed_commits), patterns)
        
        return HistoryAnalysisResult(
            repository_info=repo_info,
            commits=analyzed_commits,
            bug_predictions=bug_predictions,
            patterns=patterns,
            hotspots=hotspots,
            author_statistics=author_stats,
            timeline_analysis=timeline,
            recommendations=recommendations,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-author", response_model=AuthorAnalysisResult)
async def analyze_author(request: AuthorAnalysisRequest):
    """Analyze specific author's contribution patterns and expertise."""
    try:
        if not os.path.exists(request.repository_path):
            raise HTTPException(status_code=400, detail="Repository path does not exist")
        
        # Get author-specific commits
        author_commits = get_author_commits(request)
        
        # Analyze author statistics
        author_stats = analyze_author_statistics(author_commits)
        
        # Analyze contribution patterns
        contribution_patterns = analyze_contribution_patterns(author_commits)
        
        # Build collaboration network
        collaboration_network = build_collaboration_network(request.repository_path, request.author)
        
        # Identify expertise areas
        expertise_areas = identify_expertise_areas(author_commits)
        
        # Analyze productivity trends
        productivity_trends = analyze_productivity_trends(author_commits)
        
        # Generate recommendations
        recommendations = generate_author_recommendations(author_stats, contribution_patterns)
        
        return AuthorAnalysisResult(
            author_stats=author_stats,
            contribution_patterns=contribution_patterns,
            collaboration_network=collaboration_network,
            expertise_areas=expertise_areas,
            productivity_trends=productivity_trends,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Author analysis failed: {str(e)}")

@app.get("/repository-stats/{repository_path:path}")
async def get_repository_stats(repository_path: str):
    """Get basic repository statistics."""
    try:
        if not os.path.exists(repository_path):
            raise HTTPException(status_code=400, detail="Repository path does not exist")
        
        stats = get_basic_repository_stats(repository_path)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repository stats: {str(e)}")

# Helper Functions
def get_repository_info(repo_path: str) -> Dict[str, Any]:
    """Get basic repository information."""
    try:
        # Get remote URL
        remote_url = run_git_command(repo_path, ["config", "--get", "remote.origin.url"])
        
        # Get current branch
        current_branch = run_git_command(repo_path, ["branch", "--show-current"])
        
        # Get total commits
        total_commits = run_git_command(repo_path, ["rev-list", "--count", "HEAD"])
        
        # Get contributors
        contributors = run_git_command(repo_path, ["shortlog", "-sn", "--all"])
        
        return {
            "remote_url": remote_url.strip() if remote_url else "Unknown",
            "current_branch": current_branch.strip() if current_branch else "Unknown",
            "total_commits": int(total_commits.strip()) if total_commits else 0,
            "contributors_count": len(contributors.strip().split('\n')) if contributors else 0,
            "path": repo_path
        }
    except Exception:
        return {"path": repo_path, "error": "Failed to get repository info"}

def run_git_command(repo_path: str, args: List[str]) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""

def get_commit_history(request: HistoryAnalysisRequest) -> List[Dict[str, Any]]:
    """Get commit history based on request parameters."""
    try:
        # Build git log command
        git_args = ["log", "--pretty=format:%H|%an|%ae|%ad|%s", "--date=iso"]
        
        if request.max_commits:
            git_args.extend(["-n", str(request.max_commits)])
        
        if request.author:
            git_args.extend(["--author", request.author])
        
        if request.since_date:
            git_args.extend(["--since", request.since_date])
        
        if request.until_date:
            git_args.extend(["--until", request.until_date])
        
        if request.file_path:
            git_args.extend(["--", request.file_path])
        
        # Get commit log
        log_output = run_git_command(request.repository_path, git_args)
        
        commits = []
        for line in log_output.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|', 4)
            if len(parts) >= 5:
                commit_hash = parts[0]
                
                # Get file changes for this commit
                files_changed, insertions, deletions = get_commit_changes(request.repository_path, commit_hash)
                
                # Get diff if requested
                diff = None
                if request.include_diffs:
                    diff = get_commit_diff(request.repository_path, commit_hash)
                
                commits.append({
                    "hash": commit_hash,
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "message": parts[4],
                    "files_changed": files_changed,
                    "insertions": insertions,
                    "deletions": deletions,
                    "diff": diff
                })
        
        return commits
    except Exception:
        return []

def get_commit_changes(repo_path: str, commit_hash: str) -> tuple:
    """Get file changes, insertions, and deletions for a commit."""
    try:
        # Get file names
        files_output = run_git_command(repo_path, ["show", "--name-only", "--pretty=format:", commit_hash])
        files_changed = [f.strip() for f in files_output.strip().split('\n') if f.strip()]
        
        # Get insertions and deletions
        stats_output = run_git_command(repo_path, ["show", "--stat", "--pretty=format:", commit_hash])
        insertions, deletions = parse_commit_stats(stats_output)
        
        return files_changed, insertions, deletions
    except Exception:
        return [], 0, 0

def parse_commit_stats(stats_output: str) -> tuple:
    """Parse insertions and deletions from git stat output."""
    insertions = deletions = 0
    
    # Look for pattern like "5 files changed, 123 insertions(+), 45 deletions(-)"
    pattern = r'(\d+) insertion[s]?\(\+\)|(\d+) deletion[s]?\(\-\)'
    matches = re.findall(pattern, stats_output)
    
    for match in matches:
        if match[0]:  # insertions
            insertions += int(match[0])
        if match[1]:  # deletions
            deletions += int(match[1])
    
    return insertions, deletions

def get_commit_diff(repo_path: str, commit_hash: str) -> str:
    """Get the diff for a specific commit."""
    try:
        return run_git_command(repo_path, ["show", commit_hash])
    except Exception:
        return ""

def analyze_commits_for_bugs(commits: List[Dict[str, Any]]) -> List[CommitInfo]:
    """Analyze commits for bug indicators and calculate risk scores."""
    analyzed_commits = []
    
    # Bug indicator patterns
    bug_patterns = [
        r'\b(fix|bug|error|issue|problem|crash|fail)\b',
        r'\b(hotfix|patch|urgent)\b',
        r'\b(broken|incorrect|wrong)\b',
        r'\b(security|vulnerability|exploit)\b'
    ]
    
    for commit in commits:
        bug_indicators = []
        risk_score = 0.0
        
        # Check commit message for bug indicators
        message_lower = commit["message"].lower()
        for pattern in bug_patterns:
            if re.search(pattern, message_lower):
                bug_indicators.append(f"Message contains: {pattern}")
                risk_score += 0.3
        
        # Check file changes for risk indicators
        if len(commit["files_changed"]) > 10:
            bug_indicators.append("Large number of files changed")
            risk_score += 0.2
        
        if commit["insertions"] + commit["deletions"] > 500:
            bug_indicators.append("Large code change")
            risk_score += 0.2
        
        # Check for critical files
        critical_files = ['.py', '.js', '.ts', '.java', '.cpp', '.c']
        for file_path in commit["files_changed"]:
            if any(file_path.endswith(ext) for ext in critical_files):
                if 'main' in file_path.lower() or 'core' in file_path.lower():
                    bug_indicators.append(f"Critical file modified: {file_path}")
                    risk_score += 0.1
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        analyzed_commits.append(CommitInfo(
            hash=commit["hash"],
            author=commit["author"],
            email=commit["email"],
            date=commit["date"],
            message=commit["message"],
            files_changed=commit["files_changed"],
            insertions=commit["insertions"],
            deletions=commit["deletions"],
            diff=commit.get("diff"),
            bug_indicators=bug_indicators,
            risk_score=risk_score
        ))
    
    return analyzed_commits

def generate_bug_predictions(repo_path: str, commits: List[CommitInfo]) -> List[BugPrediction]:
    """Generate bug predictions for files based on history analysis."""
    file_stats = defaultdict(lambda: {
        'changes': 0,
        'bug_commits': 0,
        'total_risk': 0.0,
        'recent_changes': 0,
        'complexity_indicators': []
    })
    
    # Analyze file change patterns
    recent_threshold = datetime.now() - timedelta(days=30)
    
    for commit in commits:
        commit_date = datetime.fromisoformat(commit.date.replace('Z', '+00:00'))
        is_recent = commit_date > recent_threshold
        
        for file_path in commit.files_changed:
            file_stats[file_path]['changes'] += 1
            file_stats[file_path]['total_risk'] += commit.risk_score
            
            if commit.risk_score > 0.5:
                file_stats[file_path]['bug_commits'] += 1
            
            if is_recent:
                file_stats[file_path]['recent_changes'] += 1
            
            # Add complexity indicators
            if commit.insertions + commit.deletions > 100:
                file_stats[file_path]['complexity_indicators'].append('Large changes')
            
            if len(commit.files_changed) > 5:
                file_stats[file_path]['complexity_indicators'].append('Multi-file commits')
    
    # Generate predictions
    predictions = []
    for file_path, stats in file_stats.items():
        if stats['changes'] < 2:  # Skip files with minimal changes
            continue
        
        # Calculate risk score
        change_frequency_score = min(1.0, stats['changes'] / 20)
        bug_history_score = min(1.0, stats['bug_commits'] / max(1, stats['changes']))
        recent_activity_score = min(1.0, stats['recent_changes'] / 10)
        avg_risk_score = stats['total_risk'] / stats['changes']
        
        risk_score = (change_frequency_score * 0.3 + 
                     bug_history_score * 0.4 + 
                     recent_activity_score * 0.2 + 
                     avg_risk_score * 0.1)
        
        # Calculate confidence
        confidence = min(1.0, stats['changes'] / 10)
        
        # Generate factors
        factors = []
        if change_frequency_score > 0.5:
            factors.append('High change frequency')
        if bug_history_score > 0.3:
            factors.append('History of bug fixes')
        if recent_activity_score > 0.5:
            factors.append('Recent heavy activity')
        
        # Generate recommendations
        recommendations = []
        if risk_score > 0.7:
            recommendations.extend([
                'Consider additional code review',
                'Implement comprehensive testing',
                'Monitor for issues in production'
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                'Review recent changes carefully',
                'Ensure adequate test coverage'
            ])
        
        predictions.append(BugPrediction(
            file_path=file_path,
            risk_score=risk_score,
            confidence=confidence,
            factors=factors,
            recent_changes=stats['recent_changes'],
            bug_history=stats['bug_commits'],
            complexity_indicators=list(set(stats['complexity_indicators'])),
            recommendations=recommendations
        ))
    
    # Sort by risk score
    predictions.sort(key=lambda x: x.risk_score, reverse=True)
    return predictions[:20]  # Return top 20 risky files

def analyze_commit_patterns(commits: List[CommitInfo]) -> Dict[str, Any]:
    """Analyze patterns in commit history."""
    if not commits:
        return {}
    
    # Time patterns
    commit_times = []
    commit_days = []
    
    for commit in commits:
        try:
            dt = datetime.fromisoformat(commit.date.replace('Z', '+00:00'))
            commit_times.append(dt.hour)
            commit_days.append(dt.weekday())
        except Exception:
            continue
    
    # Message patterns
    message_lengths = [len(commit.message) for commit in commits]
    
    # Change size patterns
    change_sizes = [commit.insertions + commit.deletions for commit in commits]
    
    # File type patterns
    file_extensions = Counter()
    for commit in commits:
        for file_path in commit.files_changed:
            if '.' in file_path:
                ext = file_path.split('.')[-1].lower()
                file_extensions[ext] += 1
    
    return {
        'time_patterns': {
            'most_active_hours': Counter(commit_times).most_common(3),
            'most_active_days': Counter(commit_days).most_common(3)
        },
        'message_patterns': {
            'avg_length': sum(message_lengths) / len(message_lengths) if message_lengths else 0,
            'short_messages': len([l for l in message_lengths if l < 20])
        },
        'change_patterns': {
            'avg_change_size': sum(change_sizes) / len(change_sizes) if change_sizes else 0,
            'large_changes': len([s for s in change_sizes if s > 500])
        },
        'file_patterns': {
            'most_changed_types': file_extensions.most_common(5)
        }
    }

def identify_hotspots(commits: List[CommitInfo]) -> List[Dict[str, Any]]:
    """Identify code hotspots based on change frequency and risk."""
    file_hotspots = defaultdict(lambda: {'changes': 0, 'total_risk': 0.0, 'authors': set()})
    
    for commit in commits:
        for file_path in commit.files_changed:
            file_hotspots[file_path]['changes'] += 1
            file_hotspots[file_path]['total_risk'] += commit.risk_score
            file_hotspots[file_path]['authors'].add(commit.author)
    
    hotspots = []
    for file_path, stats in file_hotspots.items():
        if stats['changes'] >= 3:  # Only consider files with multiple changes
            hotspot_score = stats['changes'] * (stats['total_risk'] / stats['changes'])
            
            hotspots.append({
                'file_path': file_path,
                'change_count': stats['changes'],
                'avg_risk': stats['total_risk'] / stats['changes'],
                'hotspot_score': hotspot_score,
                'author_count': len(stats['authors']),
                'authors': list(stats['authors'])
            })
    
    # Sort by hotspot score
    hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
    return hotspots[:10]

def generate_author_statistics(commits: List[CommitInfo]) -> Dict[str, Any]:
    """Generate statistics about authors."""
    author_stats = defaultdict(lambda: {
        'commits': 0,
        'insertions': 0,
        'deletions': 0,
        'files_changed': set(),
        'avg_risk': 0.0,
        'total_risk': 0.0
    })
    
    for commit in commits:
        author = commit.author
        author_stats[author]['commits'] += 1
        author_stats[author]['insertions'] += commit.insertions
        author_stats[author]['deletions'] += commit.deletions
        author_stats[author]['files_changed'].update(commit.files_changed)
        author_stats[author]['total_risk'] += commit.risk_score
    
    # Calculate averages and convert sets to counts
    for author, stats in author_stats.items():
        stats['avg_risk'] = stats['total_risk'] / stats['commits'] if stats['commits'] > 0 else 0
        stats['files_changed'] = len(stats['files_changed'])
    
    # Sort by commit count
    sorted_authors = sorted(author_stats.items(), key=lambda x: x[1]['commits'], reverse=True)
    
    return {
        'total_authors': len(author_stats),
        'top_contributors': dict(sorted_authors[:10]),
        'collaboration_score': len(author_stats) / max(1, len(commits)) * 100
    }

def analyze_timeline(commits: List[CommitInfo]) -> Dict[str, Any]:
    """Analyze commit timeline patterns."""
    if not commits:
        return {}
    
    # Group commits by date
    daily_commits = defaultdict(int)
    daily_risk = defaultdict(float)
    
    for commit in commits:
        try:
            dt = datetime.fromisoformat(commit.date.replace('Z', '+00:00'))
            date_key = dt.date().isoformat()
            daily_commits[date_key] += 1
            daily_risk[date_key] += commit.risk_score
        except Exception:
            continue
    
    # Calculate trends
    dates = sorted(daily_commits.keys())
    if len(dates) < 2:
        return {'error': 'Insufficient data for timeline analysis'}
    
    recent_activity = sum(daily_commits[date] for date in dates[-7:]) if len(dates) >= 7 else 0
    avg_daily_commits = sum(daily_commits.values()) / len(dates)
    
    return {
        'date_range': {'start': dates[0], 'end': dates[-1]},
        'total_days': len(dates),
        'avg_daily_commits': avg_daily_commits,
        'recent_activity': recent_activity,
        'most_active_day': max(daily_commits.items(), key=lambda x: x[1]) if daily_commits else None,
        'highest_risk_day': max(daily_risk.items(), key=lambda x: x[1]) if daily_risk else None
    }

def generate_history_recommendations(commits: List[CommitInfo], predictions: List[BugPrediction], patterns: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on history analysis."""
    recommendations = []
    
    if not commits:
        return ["No commit history available for analysis"]
    
    # High-risk files
    high_risk_files = [p for p in predictions if p.risk_score > 0.7]
    if high_risk_files:
        recommendations.append(f"Monitor {len(high_risk_files)} high-risk files for potential issues")
    
    # Commit patterns
    if patterns.get('message_patterns', {}).get('short_messages', 0) > len(commits) * 0.3:
        recommendations.append("Improve commit message quality - many messages are too short")
    
    if patterns.get('change_patterns', {}).get('large_changes', 0) > len(commits) * 0.2:
        recommendations.append("Consider breaking down large commits into smaller, focused changes")
    
    # Bug indicators
    bug_commits = [c for c in commits if c.risk_score > 0.5]
    if len(bug_commits) > len(commits) * 0.3:
        recommendations.append("High frequency of bug-related commits detected - consider improving testing")
    
    # Recent activity
    recent_commits = [c for c in commits[:10]]  # Last 10 commits
    if sum(c.risk_score for c in recent_commits) / len(recent_commits) > 0.4:
        recommendations.append("Recent commits show elevated risk - consider additional review")
    
    return recommendations or ["No specific recommendations based on current analysis"]

def calculate_analysis_confidence(commit_count: int, patterns: Dict[str, Any]) -> float:
    """Calculate confidence score for the analysis."""
    base_confidence = min(1.0, commit_count / 50)  # More commits = higher confidence
    
    # Adjust based on data quality
    if patterns and 'time_patterns' in patterns:
        base_confidence += 0.1
    
    if commit_count > 100:
        base_confidence += 0.1
    
    return min(1.0, base_confidence)

# Mock implementations for author analysis functions
def get_author_commits(request: AuthorAnalysisRequest) -> List[Dict[str, Any]]:
    """Get commits for specific author analysis."""
    # This would implement author-specific commit retrieval
    return []

def analyze_author_statistics(commits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze statistics for a specific author."""
    return {"commits": len(commits), "note": "Mock implementation"}

def analyze_contribution_patterns(commits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze contribution patterns for an author."""
    return {"pattern": "regular", "note": "Mock implementation"}

def build_collaboration_network(repo_path: str, author: Optional[str]) -> Dict[str, Any]:
    """Build collaboration network for an author."""
    return {"collaborators": [], "note": "Mock implementation"}

def identify_expertise_areas(commits: List[Dict[str, Any]]) -> List[str]:
    """Identify expertise areas for an author."""
    return ["Python", "JavaScript"]

def analyze_productivity_trends(commits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze productivity trends for an author."""
    return {"trend": "stable", "note": "Mock implementation"}

def generate_author_recommendations(stats: Dict[str, Any], patterns: Dict[str, Any]) -> List[str]:
    """Generate recommendations for an author."""
    return ["Continue current contribution patterns"]

def get_basic_repository_stats(repo_path: str) -> Dict[str, Any]:
    """Get basic repository statistics."""
    try:
        total_commits = run_git_command(repo_path, ["rev-list", "--count", "HEAD"])
        total_authors = run_git_command(repo_path, ["shortlog", "-sn", "--all"])
        
        return {
            "total_commits": int(total_commits.strip()) if total_commits else 0,
            "total_authors": len(total_authors.strip().split('\n')) if total_authors else 0,
            "repository_path": repo_path
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)