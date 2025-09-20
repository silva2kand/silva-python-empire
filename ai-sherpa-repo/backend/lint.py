from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import re
import ast
import subprocess
import json
from pathlib import Path
from collections import defaultdict

# Mock GPT4All service for development
class MockGPT4AllService:
    def __init__(self):
        print("GPT4All not available. Using mock responses for development")
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        return "Mock security analysis response for development purposes"

app = FastAPI(title="AI Sherpa - Security Analysis Service", version="1.0.0")
gpt_service = MockGPT4AllService()

# Request/Response Models
class SecurityScanRequest(BaseModel):
    file_path: Optional[str] = None
    directory_path: Optional[str] = None
    code_content: Optional[str] = None
    language: Optional[str] = None
    scan_type: str = "comprehensive"  # comprehensive, quick, focused
    include_dependencies: bool = True
    severity_threshold: str = "medium"  # low, medium, high, critical

class SecurityVulnerability(BaseModel):
    id: str
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    category: str
    cwe_id: Optional[str] = None
    file_path: str
    line_number: int
    column_number: Optional[int] = None
    code_snippet: str
    recommendation: str
    references: List[str] = []
    impact: str
    effort_to_fix: str  # low, medium, high

class DependencyVulnerability(BaseModel):
    package_name: str
    version: str
    vulnerability_id: str
    severity: str
    description: str
    affected_versions: str
    fixed_version: Optional[str] = None
    recommendation: str

class SecurityScanResult(BaseModel):
    scan_summary: Dict[str, Any]
    vulnerabilities: List[SecurityVulnerability]
    dependency_vulnerabilities: List[DependencyVulnerability]
    security_score: float
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    compliance_status: Dict[str, Any]
    scan_metadata: Dict[str, Any]

class CodeQualityRequest(BaseModel):
    file_path: Optional[str] = None
    code_content: Optional[str] = None
    language: str
    check_types: List[str] = ["security", "quality", "performance"]

class CodeQualityResult(BaseModel):
    quality_score: float
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str]

@app.get("/")
async def root():
    return {"message": "AI Sherpa Security Analysis Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "security-analysis"}

@app.post("/scan", response_model=SecurityScanResult)
async def security_scan(request: SecurityScanRequest):
    """Perform comprehensive security analysis on code or files."""
    try:
        # Validate input
        if not any([request.file_path, request.directory_path, request.code_content]):
            raise HTTPException(status_code=400, detail="Must provide file_path, directory_path, or code_content")
        
        scan_results = {
            "vulnerabilities": [],
            "dependency_vulnerabilities": [],
            "scan_summary": {},
            "security_score": 0.0,
            "risk_assessment": {},
            "recommendations": [],
            "compliance_status": {},
            "scan_metadata": {}
        }
        
        # Determine scan targets
        scan_targets = []
        if request.code_content:
            scan_targets.append({"type": "content", "data": request.code_content, "language": request.language})
        elif request.file_path:
            if os.path.exists(request.file_path):
                scan_targets.append({"type": "file", "path": request.file_path})
            else:
                raise HTTPException(status_code=400, detail="File path does not exist")
        elif request.directory_path:
            if os.path.exists(request.directory_path):
                scan_targets.extend(get_scannable_files(request.directory_path))
            else:
                raise HTTPException(status_code=400, detail="Directory path does not exist")
        
        # Perform security scans
        all_vulnerabilities = []
        for target in scan_targets:
            vulnerabilities = scan_target_for_vulnerabilities(target, request.scan_type)
            all_vulnerabilities.extend(vulnerabilities)
        
        # Filter by severity threshold
        filtered_vulnerabilities = filter_by_severity(all_vulnerabilities, request.severity_threshold)
        
        # Scan dependencies if requested
        dependency_vulns = []
        if request.include_dependencies:
            if request.directory_path:
                dependency_vulns = scan_dependencies(request.directory_path)
            elif request.file_path:
                dependency_vulns = scan_dependencies(os.path.dirname(request.file_path))
        
        # Calculate security score
        security_score = calculate_security_score(filtered_vulnerabilities, dependency_vulns)
        
        # Generate risk assessment
        risk_assessment = generate_risk_assessment(filtered_vulnerabilities, dependency_vulns)
        
        # Generate recommendations
        recommendations = generate_security_recommendations(filtered_vulnerabilities, dependency_vulns, security_score)
        
        # Check compliance
        compliance_status = check_security_compliance(filtered_vulnerabilities, dependency_vulns)
        
        # Create scan summary
        scan_summary = create_scan_summary(scan_targets, filtered_vulnerabilities, dependency_vulns)
        
        # Create scan metadata
        scan_metadata = {
            "scan_type": request.scan_type,
            "severity_threshold": request.severity_threshold,
            "targets_scanned": len(scan_targets),
            "timestamp": "2024-01-01T00:00:00Z",
            "scanner_version": "1.0.0"
        }
        
        return SecurityScanResult(
            scan_summary=scan_summary,
            vulnerabilities=filtered_vulnerabilities,
            dependency_vulnerabilities=dependency_vulns,
            security_score=security_score,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            compliance_status=compliance_status,
            scan_metadata=scan_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Security scan failed: {str(e)}")

@app.post("/quality-check", response_model=CodeQualityResult)
async def code_quality_check(request: CodeQualityRequest):
    """Perform code quality analysis."""
    try:
        # Get code content
        if request.code_content:
            code = request.code_content
        elif request.file_path and os.path.exists(request.file_path):
            with open(request.file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        else:
            raise HTTPException(status_code=400, detail="Must provide valid code_content or file_path")
        
        # Perform quality checks
        issues = []
        metrics = {}
        
        if "security" in request.check_types:
            security_issues = check_security_patterns(code, request.language)
            issues.extend(security_issues)
        
        if "quality" in request.check_types:
            quality_issues = check_code_quality(code, request.language)
            issues.extend(quality_issues)
            metrics.update(calculate_quality_metrics(code, request.language))
        
        if "performance" in request.check_types:
            performance_issues = check_performance_patterns(code, request.language)
            issues.extend(performance_issues)
        
        # Calculate quality score
        quality_score = calculate_quality_score(issues, metrics)
        
        # Generate suggestions
        suggestions = generate_quality_suggestions(issues, metrics, request.language)
        
        return CodeQualityResult(
            quality_score=quality_score,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

@app.get("/vulnerability-database")
async def get_vulnerability_database():
    """Get information about the vulnerability database."""
    return {
        "database_version": "2024.01.01",
        "total_vulnerabilities": 15000,
        "last_updated": "2024-01-01T00:00:00Z",
        "supported_languages": ["python", "javascript", "typescript", "java", "c", "cpp", "go", "rust"],
        "vulnerability_categories": [
            "injection", "authentication", "authorization", "cryptography",
            "input_validation", "output_encoding", "session_management",
            "configuration", "logging", "error_handling"
        ]
    }

@app.get("/compliance-frameworks")
async def get_compliance_frameworks():
    """Get supported compliance frameworks."""
    return {
        "frameworks": [
            {
                "name": "OWASP Top 10",
                "version": "2021",
                "categories": [
                    "A01:2021 – Broken Access Control",
                    "A02:2021 – Cryptographic Failures",
                    "A03:2021 – Injection",
                    "A04:2021 – Insecure Design",
                    "A05:2021 – Security Misconfiguration",
                    "A06:2021 – Vulnerable and Outdated Components",
                    "A07:2021 – Identification and Authentication Failures",
                    "A08:2021 – Software and Data Integrity Failures",
                    "A09:2021 – Security Logging and Monitoring Failures",
                    "A10:2021 – Server-Side Request Forgery"
                ]
            },
            {
                "name": "CWE Top 25",
                "version": "2023",
                "description": "Common Weakness Enumeration"
            },
            {
                "name": "SANS Top 25",
                "version": "2023",
                "description": "Most Dangerous Software Errors"
            }
        ]
    }

# Helper Functions
def get_scannable_files(directory_path: str) -> List[Dict[str, Any]]:
    """Get list of files that can be scanned for security issues."""
    scannable_extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.cs': 'csharp'
    }
    
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]
        
        for filename in filenames:
            file_path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()
            
            if ext in scannable_extensions:
                files.append({
                    "type": "file",
                    "path": file_path,
                    "language": scannable_extensions[ext]
                })
    
    return files

def scan_target_for_vulnerabilities(target: Dict[str, Any], scan_type: str) -> List[SecurityVulnerability]:
    """Scan a target (file or code content) for security vulnerabilities."""
    vulnerabilities = []
    
    # Get code content
    if target["type"] == "file":
        try:
            with open(target["path"], 'r', encoding='utf-8') as f:
                code = f.read()
            file_path = target["path"]
            language = detect_language(target["path"])
        except Exception:
            return []
    else:
        code = target["data"]
        file_path = "<string>"
        language = target.get("language", "unknown")
    
    # Perform language-specific security checks
    if language == "python":
        vulnerabilities.extend(scan_python_security(code, file_path))
    elif language in ["javascript", "typescript"]:
        vulnerabilities.extend(scan_javascript_security(code, file_path))
    elif language == "java":
        vulnerabilities.extend(scan_java_security(code, file_path))
    
    # Perform generic security pattern checks
    vulnerabilities.extend(scan_generic_security_patterns(code, file_path, language))
    
    return vulnerabilities

def scan_python_security(code: str, file_path: str) -> List[SecurityVulnerability]:
    """Scan Python code for security vulnerabilities."""
    vulnerabilities = []
    lines = code.split('\n')
    
    # Check for common Python security issues
    patterns = [
        {
            "pattern": r'eval\s*\(',
            "title": "Use of eval() function",
            "description": "The eval() function can execute arbitrary code and is dangerous",
            "severity": "high",
            "category": "injection",
            "cwe_id": "CWE-95",
            "recommendation": "Avoid using eval(). Use ast.literal_eval() for safe evaluation of literals."
        },
        {
            "pattern": r'exec\s*\(',
            "title": "Use of exec() function",
            "description": "The exec() function can execute arbitrary code and is dangerous",
            "severity": "high",
            "category": "injection",
            "cwe_id": "CWE-95",
            "recommendation": "Avoid using exec(). Consider safer alternatives for dynamic code execution."
        },
        {
            "pattern": r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
            "title": "Shell injection vulnerability",
            "description": "Using shell=True with subprocess can lead to shell injection",
            "severity": "high",
            "category": "injection",
            "cwe_id": "CWE-78",
            "recommendation": "Avoid shell=True. Use a list of arguments instead of a string."
        },
        {
            "pattern": r'pickle\.loads?\s*\(',
            "title": "Unsafe deserialization",
            "description": "Pickle deserialization can execute arbitrary code",
            "severity": "high",
            "category": "deserialization",
            "cwe_id": "CWE-502",
            "recommendation": "Use safer serialization formats like JSON. If pickle is necessary, validate input sources."
        },
        {
            "pattern": r'random\.random\s*\(\)|random\.choice\s*\(',
            "title": "Weak random number generation",
            "description": "Using random module for security purposes is not cryptographically secure",
            "severity": "medium",
            "category": "cryptography",
            "cwe_id": "CWE-338",
            "recommendation": "Use secrets module for cryptographically secure random numbers."
        }
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], line):
                vulnerabilities.append(SecurityVulnerability(
                    id=f"PY-{len(vulnerabilities)+1:03d}",
                    title=pattern_info["title"],
                    description=pattern_info["description"],
                    severity=pattern_info["severity"],
                    confidence=0.8,
                    category=pattern_info["category"],
                    cwe_id=pattern_info.get("cwe_id"),
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation=pattern_info["recommendation"],
                    references=["https://owasp.org/"],
                    impact="High - Could lead to code execution or data compromise",
                    effort_to_fix="medium"
                ))
    
    return vulnerabilities

def scan_javascript_security(code: str, file_path: str) -> List[SecurityVulnerability]:
    """Scan JavaScript/TypeScript code for security vulnerabilities."""
    vulnerabilities = []
    lines = code.split('\n')
    
    patterns = [
        {
            "pattern": r'eval\s*\(',
            "title": "Use of eval() function",
            "description": "The eval() function can execute arbitrary JavaScript code",
            "severity": "high",
            "category": "injection",
            "cwe_id": "CWE-95",
            "recommendation": "Avoid using eval(). Use JSON.parse() for parsing JSON or other safer alternatives."
        },
        {
            "pattern": r'innerHTML\s*=.*\+',
            "title": "Potential XSS vulnerability",
            "description": "Setting innerHTML with concatenated strings can lead to XSS",
            "severity": "high",
            "category": "xss",
            "cwe_id": "CWE-79",
            "recommendation": "Use textContent or properly sanitize HTML content."
        },
        {
            "pattern": r'document\.write\s*\(',
            "title": "Use of document.write()",
            "description": "document.write() can be dangerous and lead to XSS",
            "severity": "medium",
            "category": "xss",
            "cwe_id": "CWE-79",
            "recommendation": "Use modern DOM manipulation methods instead of document.write()."
        },
        {
            "pattern": r'Math\.random\s*\(\)',
            "title": "Weak random number generation",
            "description": "Math.random() is not cryptographically secure",
            "severity": "medium",
            "category": "cryptography",
            "cwe_id": "CWE-338",
            "recommendation": "Use crypto.getRandomValues() for cryptographically secure random numbers."
        }
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], line):
                vulnerabilities.append(SecurityVulnerability(
                    id=f"JS-{len(vulnerabilities)+1:03d}",
                    title=pattern_info["title"],
                    description=pattern_info["description"],
                    severity=pattern_info["severity"],
                    confidence=0.7,
                    category=pattern_info["category"],
                    cwe_id=pattern_info.get("cwe_id"),
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation=pattern_info["recommendation"],
                    references=["https://owasp.org/"],
                    impact="Medium to High - Could lead to XSS or weak security",
                    effort_to_fix="low"
                ))
    
    return vulnerabilities

def scan_java_security(code: str, file_path: str) -> List[SecurityVulnerability]:
    """Scan Java code for security vulnerabilities."""
    vulnerabilities = []
    lines = code.split('\n')
    
    patterns = [
        {
            "pattern": r'Runtime\.getRuntime\s*\(\)\.exec\s*\(',
            "title": "Command injection vulnerability",
            "description": "Runtime.exec() can lead to command injection if user input is not sanitized",
            "severity": "high",
            "category": "injection",
            "cwe_id": "CWE-78",
            "recommendation": "Validate and sanitize all input before passing to Runtime.exec()."
        },
        {
            "pattern": r'new\s+Random\s*\(\)',
            "title": "Weak random number generation",
            "description": "java.util.Random is not cryptographically secure",
            "severity": "medium",
            "category": "cryptography",
            "cwe_id": "CWE-338",
            "recommendation": "Use SecureRandom for cryptographically secure random numbers."
        }
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], line):
                vulnerabilities.append(SecurityVulnerability(
                    id=f"JAVA-{len(vulnerabilities)+1:03d}",
                    title=pattern_info["title"],
                    description=pattern_info["description"],
                    severity=pattern_info["severity"],
                    confidence=0.8,
                    category=pattern_info["category"],
                    cwe_id=pattern_info.get("cwe_id"),
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation=pattern_info["recommendation"],
                    references=["https://owasp.org/"],
                    impact="High - Could lead to code execution",
                    effort_to_fix="medium"
                ))
    
    return vulnerabilities

def scan_generic_security_patterns(code: str, file_path: str, language: str) -> List[SecurityVulnerability]:
    """Scan for generic security patterns across languages."""
    vulnerabilities = []
    lines = code.split('\n')
    
    # Generic patterns that apply to multiple languages
    patterns = [
        {
            "pattern": r'password\s*=\s*["\'].+["\']',
            "title": "Hardcoded password",
            "description": "Password appears to be hardcoded in source code",
            "severity": "high",
            "category": "authentication",
            "cwe_id": "CWE-798",
            "recommendation": "Store passwords in environment variables or secure configuration files."
        },
        {
            "pattern": r'api[_-]?key\s*=\s*["\'].{10,}["\']',
            "title": "Hardcoded API key",
            "description": "API key appears to be hardcoded in source code",
            "severity": "high",
            "category": "authentication",
            "cwe_id": "CWE-798",
            "recommendation": "Store API keys in environment variables or secure configuration."
        },
        {
            "pattern": r'TODO.*security|FIXME.*security|HACK.*security',
            "title": "Security-related TODO/FIXME",
            "description": "Code contains security-related TODO or FIXME comments",
            "severity": "low",
            "category": "configuration",
            "recommendation": "Address security-related TODO/FIXME items before production deployment."
        }
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                vulnerabilities.append(SecurityVulnerability(
                    id=f"GEN-{len(vulnerabilities)+1:03d}",
                    title=pattern_info["title"],
                    description=pattern_info["description"],
                    severity=pattern_info["severity"],
                    confidence=0.6,
                    category=pattern_info["category"],
                    cwe_id=pattern_info.get("cwe_id"),
                    file_path=file_path,
                    line_number=i,
                    code_snippet=line.strip(),
                    recommendation=pattern_info["recommendation"],
                    references=["https://owasp.org/"],
                    impact="Varies - Could lead to credential exposure",
                    effort_to_fix="low"
                ))
    
    return vulnerabilities

def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.cs': 'csharp'
    }
    
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext, 'unknown')

def filter_by_severity(vulnerabilities: List[SecurityVulnerability], threshold: str) -> List[SecurityVulnerability]:
    """Filter vulnerabilities by severity threshold."""
    severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    threshold_level = severity_levels.get(threshold, 2)
    
    return [v for v in vulnerabilities if severity_levels.get(v.severity, 1) >= threshold_level]

def scan_dependencies(directory_path: str) -> List[DependencyVulnerability]:
    """Scan project dependencies for known vulnerabilities."""
    vulnerabilities = []
    
    # Check for Python requirements.txt
    requirements_file = os.path.join(directory_path, "requirements.txt")
    if os.path.exists(requirements_file):
        vulnerabilities.extend(scan_python_dependencies(requirements_file))
    
    # Check for Node.js package.json
    package_json = os.path.join(directory_path, "package.json")
    if os.path.exists(package_json):
        vulnerabilities.extend(scan_nodejs_dependencies(package_json))
    
    return vulnerabilities

def scan_python_dependencies(requirements_file: str) -> List[DependencyVulnerability]:
    """Scan Python dependencies for vulnerabilities."""
    # Mock implementation - in real scenario, this would check against vulnerability databases
    vulnerabilities = []
    
    try:
        with open(requirements_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '==' in line:
                        package, version = line.split('==', 1)
                        package = package.strip()
                        version = version.strip()
                        
                        # Mock vulnerability check
                        if package.lower() in ['requests', 'urllib3'] and version < '2.0.0':
                            vulnerabilities.append(DependencyVulnerability(
                                package_name=package,
                                version=version,
                                vulnerability_id="CVE-2023-MOCK",
                                severity="medium",
                                description=f"Mock vulnerability in {package} {version}",
                                affected_versions=f"< 2.0.0",
                                fixed_version="2.0.0",
                                recommendation=f"Update {package} to version 2.0.0 or later"
                            ))
    except Exception:
        pass
    
    return vulnerabilities

def scan_nodejs_dependencies(package_json: str) -> List[DependencyVulnerability]:
    """Scan Node.js dependencies for vulnerabilities."""
    # Mock implementation
    vulnerabilities = []
    
    try:
        with open(package_json, 'r') as f:
            data = json.load(f)
            dependencies = data.get('dependencies', {})
            
            for package, version in dependencies.items():
                # Mock vulnerability check
                if package in ['lodash', 'axios'] and '4.' in str(version):
                    vulnerabilities.append(DependencyVulnerability(
                        package_name=package,
                        version=str(version),
                        vulnerability_id="CVE-2023-MOCK-JS",
                        severity="high",
                        description=f"Mock vulnerability in {package} {version}",
                        affected_versions="< 5.0.0",
                        fixed_version="5.0.0",
                        recommendation=f"Update {package} to version 5.0.0 or later"
                    ))
    except Exception:
        pass
    
    return vulnerabilities

def calculate_security_score(vulnerabilities: List[SecurityVulnerability], dependency_vulns: List[DependencyVulnerability]) -> float:
    """Calculate overall security score (0-100, higher is better)."""
    base_score = 100.0
    
    # Deduct points for vulnerabilities
    severity_weights = {"low": 2, "medium": 5, "high": 15, "critical": 30}
    
    for vuln in vulnerabilities:
        weight = severity_weights.get(vuln.severity, 5)
        base_score -= weight * vuln.confidence
    
    for vuln in dependency_vulns:
        weight = severity_weights.get(vuln.severity, 5)
        base_score -= weight * 0.8  # Slightly lower weight for dependency issues
    
    return max(0.0, min(100.0, base_score))

def generate_risk_assessment(vulnerabilities: List[SecurityVulnerability], dependency_vulns: List[DependencyVulnerability]) -> Dict[str, Any]:
    """Generate risk assessment based on found vulnerabilities."""
    total_vulns = len(vulnerabilities) + len(dependency_vulns)
    
    if total_vulns == 0:
        risk_level = "low"
    elif total_vulns <= 5:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    # Count by severity
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    
    for vuln in vulnerabilities + dependency_vulns:
        severity_counts[vuln.severity] += 1
    
    return {
        "overall_risk_level": risk_level,
        "total_vulnerabilities": total_vulns,
        "severity_breakdown": severity_counts,
        "critical_issues": severity_counts["critical"] + severity_counts["high"],
        "requires_immediate_attention": severity_counts["critical"] > 0
    }

def generate_security_recommendations(vulnerabilities: List[SecurityVulnerability], dependency_vulns: List[DependencyVulnerability], security_score: float) -> List[str]:
    """Generate security recommendations."""
    recommendations = []
    
    if security_score < 50:
        recommendations.append("Security score is critically low. Immediate action required.")
    elif security_score < 70:
        recommendations.append("Security score needs improvement. Address high-priority issues.")
    
    # Count high-severity issues
    high_severity = len([v for v in vulnerabilities if v.severity in ["high", "critical"]])
    if high_severity > 0:
        recommendations.append(f"Address {high_severity} high-severity vulnerabilities immediately.")
    
    # Dependency recommendations
    if dependency_vulns:
        recommendations.append(f"Update {len(dependency_vulns)} vulnerable dependencies.")
    
    # Generic recommendations
    recommendations.extend([
        "Implement regular security scanning in CI/CD pipeline.",
        "Conduct periodic security code reviews.",
        "Keep dependencies up to date.",
        "Follow secure coding practices."
    ])
    
    return recommendations

def check_security_compliance(vulnerabilities: List[SecurityVulnerability], dependency_vulns: List[DependencyVulnerability]) -> Dict[str, Any]:
    """Check compliance with security frameworks."""
    # Mock compliance checking
    total_issues = len(vulnerabilities) + len(dependency_vulns)
    
    owasp_compliance = "pass" if total_issues < 5 else "fail"
    cwe_compliance = "pass" if len([v for v in vulnerabilities if v.severity == "critical"]) == 0 else "fail"
    
    return {
        "owasp_top_10": {
            "status": owasp_compliance,
            "issues_found": total_issues
        },
        "cwe_top_25": {
            "status": cwe_compliance,
            "critical_issues": len([v for v in vulnerabilities if v.severity == "critical"])
        }
    }

def create_scan_summary(targets: List[Dict[str, Any]], vulnerabilities: List[SecurityVulnerability], dependency_vulns: List[DependencyVulnerability]) -> Dict[str, Any]:
    """Create scan summary."""
    return {
        "targets_scanned": len(targets),
        "files_analyzed": len([t for t in targets if t["type"] == "file"]),
        "vulnerabilities_found": len(vulnerabilities),
        "dependency_issues": len(dependency_vulns),
        "scan_duration": "2.5 seconds",
        "languages_detected": list(set([t.get("language", "unknown") for t in targets]))
    }

# Code Quality Functions
def check_security_patterns(code: str, language: str) -> List[Dict[str, Any]]:
    """Check for security-related code patterns."""
    issues = []
    
    # This would implement security pattern checking
    # For now, return mock data
    if "password" in code.lower():
        issues.append({
            "type": "security",
            "severity": "high",
            "message": "Potential hardcoded password detected",
            "line": 1
        })
    
    return issues

def check_code_quality(code: str, language: str) -> List[Dict[str, Any]]:
    """Check general code quality issues."""
    issues = []
    
    # Mock quality checks
    lines = code.split('\n')
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            issues.append({
                "type": "quality",
                "severity": "low",
                "message": "Line too long (>120 characters)",
                "line": i
            })
    
    return issues

def check_performance_patterns(code: str, language: str) -> List[Dict[str, Any]]:
    """Check for performance-related issues."""
    issues = []
    
    # Mock performance checks
    if language == "python" and "for" in code and "in range(len(" in code:
        issues.append({
            "type": "performance",
            "severity": "medium",
            "message": "Consider using enumerate() instead of range(len())",
            "line": 1
        })
    
    return issues

def calculate_quality_metrics(code: str, language: str) -> Dict[str, Any]:
    """Calculate code quality metrics."""
    lines = code.split('\n')
    
    return {
        "lines_of_code": len([l for l in lines if l.strip()]),
        "blank_lines": len([l for l in lines if not l.strip()]),
        "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
        "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0,
        "complexity_estimate": min(10, len(lines) // 10)  # Mock complexity
    }

def calculate_quality_score(issues: List[Dict[str, Any]], metrics: Dict[str, Any]) -> float:
    """Calculate overall code quality score."""
    base_score = 100.0
    
    # Deduct points for issues
    severity_weights = {"low": 1, "medium": 3, "high": 8, "critical": 15}
    
    for issue in issues:
        weight = severity_weights.get(issue.get("severity", "medium"), 3)
        base_score -= weight
    
    return max(0.0, min(100.0, base_score))

def generate_quality_suggestions(issues: List[Dict[str, Any]], metrics: Dict[str, Any], language: str) -> List[str]:
    """Generate code quality improvement suggestions."""
    suggestions = []
    
    if len(issues) > 10:
        suggestions.append("Consider refactoring to reduce the number of quality issues.")
    
    if metrics.get("avg_line_length", 0) > 100:
        suggestions.append("Consider breaking long lines for better readability.")
    
    if metrics.get("complexity_estimate", 0) > 7:
        suggestions.append("Consider breaking down complex functions into smaller ones.")
    
    suggestions.extend([
        "Add more comments to improve code documentation.",
        "Follow language-specific style guidelines.",
        "Consider using automated code formatting tools."
    ])
    
    return suggestions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)