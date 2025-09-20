"""Code Analysis Agent for AI Sherpa Multi-Agent System"""

import ast
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent
from datetime import datetime

class CodeAnalysisAgent(BaseAgent):
    """Specialized agent for code analysis, pattern recognition, and optimization"""
    
    def __init__(self):
        super().__init__(
            agent_id="code_analysis_agent_001",
            name="CodeAnalysisAgent",
            capabilities=[
                "code_parsing",
                "syntax_analysis",
                "pattern_recognition",
                "code_quality_assessment",
                "performance_analysis",
                "security_analysis",
                "refactoring_suggestions",
                "dependency_analysis",
                "complexity_measurement",
                "code_smell_detection"
            ]
        )
        self.supported_languages = [
            "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust"
        ]
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if this agent can handle the given task type"""
        analysis_tasks = [
            "code_analysis",
            "syntax_check",
            "pattern_detection",
            "quality_assessment",
            "performance_analysis",
            "security_scan",
            "refactoring_suggestions",
            "dependency_check",
            "complexity_analysis",
            "code_review",
            "optimization_suggestions"
        ]
        return task_type.lower() in analysis_tasks
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process code analysis tasks"""
        task_type = task.get('type', '').lower()
        code = task.get('code', '')
        language = task.get('language', 'python').lower()
        context = task.get('context', {})
        
        if task_type == "code_analysis":
            return await self._analyze_code_comprehensive(code, language, context)
        elif task_type == "syntax_check":
            return await self._check_syntax(code, language)
        elif task_type == "pattern_detection":
            return await self._detect_patterns(code, language, context)
        elif task_type == "quality_assessment":
            return await self._assess_code_quality(code, language)
        elif task_type == "performance_analysis":
            return await self._analyze_performance(code, language)
        elif task_type == "security_scan":
            return await self._scan_security_issues(code, language)
        elif task_type == "refactoring_suggestions":
            return await self._suggest_refactoring(code, language)
        elif task_type == "dependency_check":
            return await self._analyze_dependencies(code, language)
        elif task_type == "complexity_analysis":
            return await self._analyze_complexity(code, language)
        elif task_type == "optimization_suggestions":
            return await self._suggest_optimizations(code, language)
        else:
            return await self._general_code_analysis(code, language, context)
    
    async def _analyze_code_comprehensive(self, code: str, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive code analysis"""
        try:
            analysis_results = {
                "syntax_analysis": await self._check_syntax(code, language),
                "quality_assessment": await self._assess_code_quality(code, language),
                "performance_analysis": await self._analyze_performance(code, language),
                "security_analysis": await self._scan_security_issues(code, language),
                "complexity_analysis": await self._analyze_complexity(code, language),
                "pattern_analysis": await self._detect_patterns(code, language, context)
            }
            
            # Calculate overall score
            scores = []
            for analysis in analysis_results.values():
                if isinstance(analysis, dict) and 'score' in analysis:
                    scores.append(analysis['score'])
            
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            return {
                "analysis_type": "comprehensive_code_analysis",
                "language": language,
                "code_length": len(code),
                "overall_score": round(overall_score, 2),
                "detailed_analysis": analysis_results,
                "summary": self._generate_analysis_summary(analysis_results),
                "recommendations": self._generate_recommendations(analysis_results),
                "confidence": 0.90
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            return {
                "analysis_type": "comprehensive_code_analysis",
                "error": str(e),
                "confidence": 0.0
            }
    
    async def _check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Check code syntax"""
        try:
            if language == "python":
                return await self._check_python_syntax(code)
            elif language in ["javascript", "typescript"]:
                return await self._check_js_syntax(code)
            else:
                return await self._check_generic_syntax(code, language)
                
        except Exception as e:
            return {
                "syntax_valid": False,
                "errors": [str(e)],
                "score": 0.0
            }
    
    async def _check_python_syntax(self, code: str) -> Dict[str, Any]:
        """Check Python syntax specifically"""
        try:
            ast.parse(code)
            return {
                "syntax_valid": True,
                "errors": [],
                "warnings": [],
                "score": 10.0,
                "language": "python"
            }
        except SyntaxError as e:
            return {
                "syntax_valid": False,
                "errors": [{
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "type": "SyntaxError"
                }],
                "score": 0.0,
                "language": "python"
            }
    
    async def _check_js_syntax(self, code: str) -> Dict[str, Any]:
        """Check JavaScript/TypeScript syntax"""
        # Mock JS syntax check - in production, use a proper JS parser
        common_js_errors = [
            r'\bfunction\s+\w+\s*\([^)]*\)\s*{[^}]*$',  # Unclosed function
            r'\bif\s*\([^)]*\)\s*{[^}]*$',  # Unclosed if statement
            r'[^;]\s*$'  # Missing semicolon (simplified)
        ]
        
        errors = []
        for i, line in enumerate(code.split('\n'), 1):
            for pattern in common_js_errors:
                if re.search(pattern, line):
                    errors.append({
                        "line": i,
                        "message": "Potential syntax issue detected",
                        "type": "SyntaxWarning"
                    })
        
        return {
            "syntax_valid": len(errors) == 0,
            "errors": errors,
            "score": 10.0 if len(errors) == 0 else max(0, 10 - len(errors) * 2),
            "language": "javascript"
        }
    
    async def _check_generic_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Generic syntax check for other languages"""
        # Basic checks for common syntax issues
        lines = code.split('\n')
        errors = []
        
        # Check for basic bracket matching
        bracket_stack = []
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        
        for i, line in enumerate(lines, 1):
            for char in line:
                if char in bracket_pairs:
                    bracket_stack.append((char, i))
                elif char in bracket_pairs.values():
                    if not bracket_stack:
                        errors.append({
                            "line": i,
                            "message": f"Unmatched closing bracket: {char}",
                            "type": "SyntaxError"
                        })
                    else:
                        open_bracket, _ = bracket_stack.pop()
                        if bracket_pairs[open_bracket] != char:
                            errors.append({
                                "line": i,
                                "message": f"Mismatched brackets: {open_bracket} and {char}",
                                "type": "SyntaxError"
                            })
        
        # Check for unclosed brackets
        for bracket, line_num in bracket_stack:
            errors.append({
                "line": line_num,
                "message": f"Unclosed bracket: {bracket}",
                "type": "SyntaxError"
            })
        
        return {
            "syntax_valid": len(errors) == 0,
            "errors": errors,
            "score": 10.0 if len(errors) == 0 else max(0, 10 - len(errors)),
            "language": language
        }
    
    async def _assess_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Assess overall code quality"""
        lines = code.split('\n')
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')])
        
        # Calculate metrics
        comment_ratio = comment_lines / non_empty_lines if non_empty_lines > 0 else 0
        avg_line_length = sum(len(line) for line in lines) / total_lines if total_lines > 0 else 0
        
        # Quality indicators
        quality_issues = []
        quality_score = 10.0
        
        # Check line length
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            quality_issues.append({
                "type": "long_lines",
                "severity": "medium",
                "message": f"Lines too long (>120 chars): {long_lines[:5]}{'...' if len(long_lines) > 5 else ''}",
                "count": len(long_lines)
            })
            quality_score -= min(2.0, len(long_lines) * 0.1)
        
        # Check comment ratio
        if comment_ratio < 0.1:
            quality_issues.append({
                "type": "low_comments",
                "severity": "medium",
                "message": "Low comment ratio - consider adding more documentation",
                "ratio": round(comment_ratio, 2)
            })
            quality_score -= 1.0
        
        # Check for potential code smells
        code_smells = await self._detect_code_smells(code, language)
        if code_smells:
            quality_issues.extend(code_smells)
            quality_score -= len(code_smells) * 0.5
        
        return {
            "total_lines": total_lines,
            "non_empty_lines": non_empty_lines,
            "comment_lines": comment_lines,
            "comment_ratio": round(comment_ratio, 2),
            "avg_line_length": round(avg_line_length, 1),
            "quality_issues": quality_issues,
            "score": max(0.0, round(quality_score, 1)),
            "grade": self._calculate_grade(max(0.0, quality_score))
        }
    
    async def _detect_code_smells(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Detect common code smells"""
        smells = []
        lines = code.split('\n')
        
        # Long method detection
        if language == "python":
            in_function = False
            function_start = 0
            function_lines = 0
            
            for i, line in enumerate(lines):
                if re.match(r'^\s*def\s+\w+', line):
                    if in_function and function_lines > 50:
                        smells.append({
                            "type": "long_method",
                            "severity": "high",
                            "message": f"Method starting at line {function_start} is too long ({function_lines} lines)",
                            "line": function_start
                        })
                    in_function = True
                    function_start = i + 1
                    function_lines = 0
                elif in_function:
                    if line.strip() and not line.startswith(' '):
                        if function_lines > 50:
                            smells.append({
                                "type": "long_method",
                                "severity": "high",
                                "message": f"Method starting at line {function_start} is too long ({function_lines} lines)",
                                "line": function_start
                            })
                        in_function = False
                    else:
                        function_lines += 1
        
        # Duplicate code detection (simplified)
        line_counts = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Only check substantial lines
                if stripped in line_counts:
                    line_counts[stripped].append(i + 1)
                else:
                    line_counts[stripped] = [i + 1]
        
        for line_content, line_numbers in line_counts.items():
            if len(line_numbers) > 2:
                smells.append({
                    "type": "duplicate_code",
                    "severity": "medium",
                    "message": f"Duplicate line found at lines: {line_numbers}",
                    "content": line_content[:50] + "..." if len(line_content) > 50 else line_content
                })
        
        return smells
    
    async def _analyze_performance(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code performance characteristics"""
        performance_issues = []
        performance_score = 10.0
        
        # Check for common performance anti-patterns
        if language == "python":
            # Check for inefficient loops
            if re.search(r'for\s+\w+\s+in\s+range\(len\(', code):
                performance_issues.append({
                    "type": "inefficient_loop",
                    "severity": "medium",
                    "message": "Consider using enumerate() instead of range(len())",
                    "suggestion": "Use 'for i, item in enumerate(items):' instead of 'for i in range(len(items)):'"
                })
                performance_score -= 1.0
            
            # Check for string concatenation in loops
            if re.search(r'\+=.*["\']', code) and 'for' in code:
                performance_issues.append({
                    "type": "string_concatenation_in_loop",
                    "severity": "high",
                    "message": "String concatenation in loops is inefficient",
                    "suggestion": "Use join() method or f-strings for better performance"
                })
                performance_score -= 2.0
        
        # Check for nested loops
        nested_loop_count = len(re.findall(r'for.*for', code, re.DOTALL))
        if nested_loop_count > 0:
            performance_issues.append({
                "type": "nested_loops",
                "severity": "medium",
                "message": f"Found {nested_loop_count} nested loop(s) - consider optimization",
                "suggestion": "Consider using more efficient algorithms or data structures"
            })
            performance_score -= nested_loop_count * 0.5
        
        return {
            "performance_issues": performance_issues,
            "nested_loops": nested_loop_count,
            "score": max(0.0, round(performance_score, 1)),
            "optimization_potential": "high" if performance_score < 7 else "medium" if performance_score < 9 else "low"
        }
    
    async def _scan_security_issues(self, code: str, language: str) -> Dict[str, Any]:
        """Scan for potential security issues"""
        security_issues = []
        security_score = 10.0
        
        # Common security patterns to check
        security_patterns = {
            "sql_injection": r'(SELECT|INSERT|UPDATE|DELETE).*\+.*["\']',
            "hardcoded_password": r'(password|pwd|pass)\s*=\s*["\'][^"\']+',
            "eval_usage": r'\beval\s*\(',
            "shell_injection": r'(os\.system|subprocess\.call).*\+',
            "xss_vulnerability": r'innerHTML\s*=.*\+'
        }
        
        for issue_type, pattern in security_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                security_issues.append({
                    "type": issue_type,
                    "severity": "high" if issue_type in ["sql_injection", "shell_injection"] else "medium",
                    "message": f"Potential {issue_type.replace('_', ' ')} vulnerability detected",
                    "line": line_num,
                    "code_snippet": match.group()[:50]
                })
                security_score -= 2.0 if issue_type in ["sql_injection", "shell_injection"] else 1.0
        
        return {
            "security_issues": security_issues,
            "score": max(0.0, round(security_score, 1)),
            "risk_level": "high" if security_score < 6 else "medium" if security_score < 8 else "low"
        }
    
    async def _analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        lines = code.split('\n')
        
        # Cyclomatic complexity (simplified)
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
        complexity_score = 1  # Base complexity
        
        for line in lines:
            for keyword in complexity_keywords:
                complexity_score += line.lower().count(keyword)
        
        # Nesting depth
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent_level = (len(line) - len(stripped)) // 4  # Assuming 4-space indentation
                max_nesting = max(max_nesting, indent_level)
        
        return {
            "cyclomatic_complexity": complexity_score,
            "max_nesting_depth": max_nesting,
            "complexity_grade": self._get_complexity_grade(complexity_score),
            "score": max(0.0, 10.0 - (complexity_score * 0.1) - (max_nesting * 0.5)),
            "recommendations": self._get_complexity_recommendations(complexity_score, max_nesting)
        }
    
    def _get_complexity_grade(self, complexity: int) -> str:
        """Get complexity grade based on cyclomatic complexity"""
        if complexity <= 10:
            return "A (Low)"
        elif complexity <= 20:
            return "B (Moderate)"
        elif complexity <= 50:
            return "C (High)"
        else:
            return "D (Very High)"
    
    def _get_complexity_recommendations(self, complexity: int, nesting: int) -> List[str]:
        """Get recommendations based on complexity metrics"""
        recommendations = []
        
        if complexity > 20:
            recommendations.append("Consider breaking down complex functions into smaller ones")
        if nesting > 4:
            recommendations.append("Reduce nesting depth by extracting methods or using early returns")
        if complexity > 50:
            recommendations.append("This code is very complex - major refactoring recommended")
        
        return recommendations
    
    async def _detect_patterns(self, code: str, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect design patterns and code patterns"""
        patterns_found = []
        
        # Singleton pattern
        if re.search(r'class\s+\w+.*__new__', code, re.DOTALL):
            patterns_found.append({
                "pattern": "Singleton",
                "type": "creational",
                "confidence": 0.8,
                "description": "Singleton pattern implementation detected"
            })
        
        # Factory pattern
        if re.search(r'def\s+create_\w+|def\s+make_\w+', code):
            patterns_found.append({
                "pattern": "Factory",
                "type": "creational",
                "confidence": 0.7,
                "description": "Factory pattern methods detected"
            })
        
        # Observer pattern
        if re.search(r'(notify|subscribe|observer)', code, re.IGNORECASE):
            patterns_found.append({
                "pattern": "Observer",
                "type": "behavioral",
                "confidence": 0.6,
                "description": "Observer pattern elements detected"
            })
        
        return {
            "patterns_found": patterns_found,
            "pattern_count": len(patterns_found),
            "score": min(10.0, len(patterns_found) * 2.0),  # Bonus for using patterns
            "suggestions": self._suggest_patterns(code, language)
        }
    
    def _suggest_patterns(self, code: str, language: str) -> List[str]:
        """Suggest design patterns that might be applicable"""
        suggestions = []
        
        # Check if Strategy pattern might be useful
        if code.count('if') > 5:
            suggestions.append("Consider using Strategy pattern to replace complex conditional logic")
        
        # Check if Factory pattern might be useful
        if re.search(r'\w+\s*=\s*\w+\(', code) and code.count('=') > 3:
            suggestions.append("Consider using Factory pattern for object creation")
        
        return suggestions
    
    async def _suggest_refactoring(self, code: str, language: str) -> Dict[str, Any]:
        """Suggest refactoring opportunities"""
        suggestions = []
        
        # Long method refactoring
        lines = code.split('\n')
        if len(lines) > 50:
            suggestions.append({
                "type": "extract_method",
                "priority": "high",
                "description": "Method is too long - consider extracting smaller methods",
                "benefit": "Improved readability and maintainability"
            })
        
        # Duplicate code refactoring
        if "TODO" in code or "FIXME" in code:
            suggestions.append({
                "type": "address_todos",
                "priority": "medium",
                "description": "Address TODO and FIXME comments",
                "benefit": "Complete implementation and fix known issues"
            })
        
        return {
            "refactoring_suggestions": suggestions,
            "priority_order": sorted(suggestions, key=lambda x: x.get('priority', 'low'), reverse=True),
            "estimated_effort": "medium" if len(suggestions) > 2 else "low"
        }
    
    async def _analyze_dependencies(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code dependencies"""
        dependencies = []
        
        if language == "python":
            # Find import statements
            import_pattern = r'^\s*(import|from)\s+([\w\.]+)'
            imports = re.findall(import_pattern, code, re.MULTILINE)
            dependencies = [imp[1] for imp in imports]
        
        elif language in ["javascript", "typescript"]:
            # Find require/import statements
            require_pattern = r'require\(["\']([^"\']*)'
            import_pattern = r'import.*from\s+["\']([^"\']*)'
            requires = re.findall(require_pattern, code)
            imports = re.findall(import_pattern, code)
            dependencies = requires + imports
        
        return {
            "dependencies": dependencies,
            "dependency_count": len(dependencies),
            "external_dependencies": [dep for dep in dependencies if not dep.startswith('.')],
            "internal_dependencies": [dep for dep in dependencies if dep.startswith('.')],
            "complexity_score": min(10.0, len(dependencies) * 0.5)
        }
    
    async def _suggest_optimizations(self, code: str, language: str) -> Dict[str, Any]:
        """Suggest code optimizations"""
        optimizations = []
        
        # Performance optimizations
        perf_analysis = await self._analyze_performance(code, language)
        for issue in perf_analysis.get('performance_issues', []):
            optimizations.append({
                "type": "performance",
                "category": issue['type'],
                "description": issue['message'],
                "suggestion": issue.get('suggestion', 'Consider optimization'),
                "impact": "high" if issue['severity'] == "high" else "medium"
            })
        
        # Memory optimizations
        if 'list(' in code or 'dict(' in code:
            optimizations.append({
                "type": "memory",
                "category": "data_structures",
                "description": "Consider using generators or more efficient data structures",
                "suggestion": "Use generators for large datasets to reduce memory usage",
                "impact": "medium"
            })
        
        return {
            "optimizations": optimizations,
            "optimization_count": len(optimizations),
            "potential_impact": "high" if any(opt['impact'] == 'high' for opt in optimizations) else "medium"
        }
    
    async def _general_code_analysis(self, code: str, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general code analysis when specific type is not specified"""
        return {
            "analysis_type": "general_analysis",
            "language": language,
            "code_length": len(code),
            "line_count": len(code.split('\n')),
            "basic_metrics": {
                "functions": len(re.findall(r'def\s+\w+', code)) if language == 'python' else len(re.findall(r'function\s+\w+', code)),
                "classes": len(re.findall(r'class\s+\w+', code)),
                "comments": len(re.findall(r'#.*|//.*', code))
            },
            "summary": f"Analyzed {len(code)} characters of {language} code",
            "confidence": 0.70
        }
    
    def _generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a summary of all analysis results"""
        summaries = []
        
        for analysis_type, result in analysis_results.items():
            if isinstance(result, dict) and 'score' in result:
                score = result['score']
                if score >= 8:
                    summaries.append(f"{analysis_type}: Excellent ({score}/10)")
                elif score >= 6:
                    summaries.append(f"{analysis_type}: Good ({score}/10)")
                elif score >= 4:
                    summaries.append(f"{analysis_type}: Fair ({score}/10)")
                else:
                    summaries.append(f"{analysis_type}: Needs improvement ({score}/10)")
        
        return "; ".join(summaries)
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check each analysis for specific recommendations
        for analysis_type, result in analysis_results.items():
            if isinstance(result, dict):
                if 'quality_issues' in result and result['quality_issues']:
                    recommendations.append(f"Address {len(result['quality_issues'])} code quality issues")
                
                if 'performance_issues' in result and result['performance_issues']:
                    recommendations.append(f"Optimize {len(result['performance_issues'])} performance bottlenecks")
                
                if 'security_issues' in result and result['security_issues']:
                    recommendations.append(f"Fix {len(result['security_issues'])} security vulnerabilities")
        
        if not recommendations:
            recommendations.append("Code analysis completed - no major issues found")
        
        return recommendations
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from numeric score"""
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B+"
        elif score >= 6:
            return "B"
        elif score >= 5:
            return "C+"
        elif score >= 4:
            return "C"
        elif score >= 3:
            return "D"
        else:
            return "F"