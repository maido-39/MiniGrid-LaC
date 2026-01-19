"""
VLM Action Uncertainty Estimation

OpenAI API의 logprobs 기능을 활용하여 action 예측의 불확실도를 측정하고 시각화합니다.


주의 : malfunction now 
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math

# Actual path: legacy.vlm_rels.minigrid_vlm_controller
from legacy import MiniGridVLMController
# Actual paths: utils.map_manager.minigrid_customenv_emoji, utils.vlm.vlm_wrapper, utils.vlm.vlm_postprocessor
from utils import MiniGridEmojiWrapper, ChatGPT4oVLMWrapper, VLMResponsePostProcessor

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai 라이브러리가 필요합니다: pip install openai")


class UncertaintyVLMWrapper(ChatGPT4oVLMWrapper):
    """
    logprobs를 지원하는 VLM Wrapper
    
    기존 ChatGPT4oVLMWrapper를 확장하여 logprobs 정보를 얻을 수 있게 합니다.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        logprobs: bool = True,
        top_logprobs: int = 5
    ):
        """
        Args:
            logprobs: logprobs 반환 여부
            top_logprobs: 상위 N개 토큰의 logprobs 반환
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        
        # OpenAI 클라이언트 직접 접근 (handler를 통해)
        if hasattr(self._handler, 'client'):
            self.client = self._handler.client
        else:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate_with_logprobs(
        self,
        image: Optional[Union[str, np.ndarray]] = None,
        system_prompt: str = "",
        user_prompt: str = "",
        n: int = 1
    ) -> Union[Tuple[str, Optional[Dict]], List[Tuple[str, Optional[Dict]]]]:
        """
        logprobs 정보와 함께 응답 생성
        
        Args:
            n: 생성할 샘플 수 (1이면 단일 샘플, 1보다 크면 여러 샘플)
        
        Returns:
            n=1일 때: (response_text, logprobs_info)
            n>1일 때: [(response_text, logprobs_info), ...] 리스트
            logprobs_info: {
                'content': [{'token': str, 'logprob': float, 'top_logprobs': [...]}, ...],
                'finish_reason': str
            }
        """
        from vlm.handlers.openai_handler import OpenAIHandler
        
        # 메시지 생성 (handler의 메서드 활용)
        if hasattr(self._handler, '_build_messages'):
            messages = self._handler._build_messages(image, system_prompt, user_prompt)
        else:
            # 직접 메시지 생성
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            content = [{"type": "text", "text": user_prompt}]
            if image is not None:
                if isinstance(image, str):
                    import base64
                    with open(image, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                else:
                    # numpy array나 PIL Image는 handler의 encode_image 사용
                    image_b64 = self._encode_image(image)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    })
            
            messages.append({"role": "user", "content": content})
        
        # API 호출 (logprobs 포함)
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs if self.logprobs else None,
            "n": n  # 여러 샘플 생성
        }
        
        if self.is_gpt5_model:
            api_params["max_completion_tokens"] = self.max_tokens
        else:
            api_params["max_tokens"] = self.max_tokens
        
        if api_params["top_logprobs"] is None:
            del api_params["top_logprobs"]
        
        try:
            response = self.client.chat.completions.create(**api_params)
            
            # 여러 샘플 처리
            results = []
            for choice in response.choices:
                response_text = choice.message.content
                
                # logprobs 정보 추출
                logprobs_info = None
                if self.logprobs and hasattr(choice, 'logprobs') and choice.logprobs:
                    logprobs_info = {
                        'content': choice.logprobs.content,
                        'finish_reason': choice.finish_reason
                    }
                
                results.append((response_text, logprobs_info))
            
            # n=1이면 단일 튜플 반환, n>1이면 리스트 반환
            if n == 1:
                return results[0]
            else:
                return results
            
        except Exception as e:
            raise RuntimeError(f"API 호출 중 오류 발생: {e}")


class UncertaintyController(MiniGridVLMController):
    """
    불확실도 측정 기능이 추가된 VLM Controller
    
    기존 MiniGridVLMController를 확장하여 action 예측의 불확실도를 측정합니다.
    """
    
    def __init__(
        self,
        env: MiniGridEmojiWrapper,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        required_fields: Optional[list] = None,
        logprobs: bool = True,
        top_logprobs: int = 5,
        num_samples: int = 5,
        sampling_temperature: float = 1.0
    ):
        """
        Args:
            logprobs: logprobs 반환 여부
            top_logprobs: 상위 N개 토큰의 logprobs 반환
            num_samples: Semantic Entropy 계산을 위한 샘플링 개수 (기본값: 5)
            sampling_temperature: 샘플링 시 사용할 temperature (기본값: 1.0)
        """
        # 부모 클래스 초기화 (vlm은 나중에 교체)
        self.env = env
        self.postprocessor = VLMResponsePostProcessor(
            required_fields=required_fields or ["action", "environment_info"]
        )
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.user_prompt_template = user_prompt_template or self._get_default_user_prompt_template()
        
        # logprobs 지원 VLM 사용
        self.vlm = UncertaintyVLMWrapper(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs
        )
        
        # Semantic Entropy 계산을 위한 파라미터
        self.num_samples = num_samples
        self.sampling_temperature = sampling_temperature
    
    def generate_action_with_uncertainty(
        self,
        user_prompt: Optional[str] = None,
        mission: Optional[str] = None
    ) -> Dict:
        """
        불확실도 정보와 함께 액션 생성 (Semantic Entropy 방식)
        
        Returns:
            {
                'action': str,
                'environment_info': str,
                'reasoning': str,
                'uncertainty': {
                    'action_entropy': float,  # Semantic Entropy
                    'action_prob': float,     # 선택된 action 클러스터의 확률
                    'clusters': List[Dict],   # 의미적 클러스터 정보
                    'num_unique_actions': int  # 고유한 action 개수
                }
            }
        """
        image = self.env.get_image()
        
        if user_prompt is None:
            user_prompt = self.get_user_prompt(mission=mission)
        
        # 여러 샘플 생성 (Semantic Entropy 계산용)
        try:
            # 샘플링을 위해 temperature를 높여서 여러 샘플 생성
            original_temp = self.vlm.temperature
            self.vlm.temperature = self.sampling_temperature
            
            samples = self.vlm.generate_with_logprobs(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                n=self.num_samples
            )
            
            # temperature 원복
            self.vlm.temperature = original_temp
            
            # n=1일 때도 리스트로 변환
            if not isinstance(samples, list):
                samples = [samples]
                
        except Exception as e:
            raise RuntimeError(f"VLM API 호출 실패: {e}")
        
        # 첫 번째 샘플을 메인 응답으로 사용
        response_text, _ = samples[0]
        
        try:
            vlm_response = self.postprocessor.process(response_text, strict=True)
        except ValueError as e:
            raise ValueError(f"VLM 응답 파싱 실패: {e}\n원본 응답: {response_text[:200]}...")
        
        # Semantic Entropy 계산
        uncertainty_info = self._calculate_semantic_entropy(samples, vlm_response.get('action', ''))
        
        vlm_response['uncertainty'] = uncertainty_info
        
        return vlm_response
    
    def _check_action_equivalence(self, action1: str, action2: str) -> bool:
        """
        두 action이 의미적으로 동일한지 확인
        
        Args:
            action1: 첫 번째 action 문자열
            action2: 두 번째 action 문자열
        
        Returns:
            의미적으로 동일하면 True, 다르면 False
        """
        # 정규화: 소문자 변환 및 공백 제거
        a1 = action1.strip().lower()
        a2 = action2.strip().lower()
        
        # 완전히 동일하면 True
        if a1 == a2:
            return True
        
        # 동의어 매핑 (예: "up" = "north", "forward" = "up" 등)
        synonyms = {
            'up': ['north', 'forward', '↑'],
            'down': ['south', 'backward', '↓'],
            'left': ['west', '←'],
            'right': ['east', '→'],
            'pickup': ['pick', 'grab', 'take'],
            'drop': ['put', 'place'],
            'toggle': ['interact', 'activate']
        }
        
        # 동의어 체크
        for key, values in synonyms.items():
            if a1 == key and a2 in values:
                return True
            if a2 == key and a1 in values:
                return True
            if a1 in values and a2 in values:
                return True
        
        # 간단한 문자열 유사도 체크 (Levenshtein distance 기반)
        # 매우 짧은 action의 경우 완전 일치만 허용
        if len(a1) <= 3 and len(a2) <= 3:
            return False
        
        # VLM을 사용한 의미적 동등성 체크 (선택적, 비용이 있음)
        # 여기서는 간단한 휴리스틱만 사용
        return False
    
    def _get_sequence_log_prob(self, logprobs_content) -> float:
        """
        전체 시퀀스의 log probability 합 계산
        
        Args:
            logprobs_content: logprobs 정보의 content 리스트
        
        Returns:
            전체 시퀀스의 logprob 합
        """
        if logprobs_content is None:
            return 0.0
        
        sum_logprob = 0.0
        for token_data in logprobs_content:
            sum_logprob += token_data.logprob
        return sum_logprob
    
    def _calculate_semantic_entropy(
        self,
        samples: List[Tuple[str, Optional[Dict]]],
        main_action: str = ""
    ) -> Dict:
        """
        Semantic Entropy 방식으로 불확실도 계산
        
        여러 샘플을 생성하고, 의미적으로 동일한 action들을 클러스터링한 후
        클러스터 간 확률 분포로 엔트로피를 계산합니다.
        
        Args:
            samples: [(response_text, logprobs_info), ...] 샘플 리스트
            main_action: 메인 응답의 action (선택적)
        
        Returns:
            {
                'action_entropy': float,  # Semantic Entropy
                'action_prob': float,     # 선택된 action 클러스터의 확률
                'clusters': List[Dict],   # 클러스터 정보
                'num_unique_actions': int  # 고유한 action 개수
            }
        """
        if not samples:
            return {
                'action_entropy': None,
                'action_prob': None,
                'clusters': [],
                'num_unique_actions': 0
            }
        
        # 각 샘플에서 action 추출 및 확률 계산
        sample_data = []
        for response_text, logprobs_info in samples:
            try:
                # JSON에서 action 추출
                parsed = self.postprocessor.process(response_text, strict=False)
                action = parsed.get('action', '').strip()
                
                # 전체 시퀀스의 logprob 계산
                if logprobs_info and logprobs_info.get('content'):
                    logprob = self._get_sequence_log_prob(logprobs_info['content'])
                    prob = math.exp(logprob)
                else:
                    logprob = 0.0
                    prob = 1.0 / len(samples)  # logprob가 없으면 균등 분포 가정
                
                if action:
                    sample_data.append({
                        'action': action,
                        'text': response_text,
                        'logprob': logprob,
                        'prob': prob
                    })
            except Exception as e:
                # 파싱 실패한 샘플은 건너뛰기
                continue
        
        if not sample_data:
            return {
                'action_entropy': None,
                'action_prob': None,
                'clusters': [],
                'num_unique_actions': 0
            }
        
        # 의미적 클러스터링
        clusters = []
        for sample in sample_data:
            matched = False
            for cluster in clusters:
                # 클러스터의 첫 번째 action과 비교
                if self._check_action_equivalence(cluster['actions'][0], sample['action']):
                    cluster['actions'].append(sample['action'])
                    cluster['texts'].append(sample['text'])
                    cluster['prob_sum'] += sample['prob']
                    matched = True
                    break
            
            if not matched:
                clusters.append({
                    'actions': [sample['action']],
                    'texts': [sample['text']],
                    'prob_sum': sample['prob']
                })
        
        # 확률 정규화
        total_prob = sum(c['prob_sum'] for c in clusters)
        if total_prob > 0:
            for cluster in clusters:
                cluster['probability'] = cluster['prob_sum'] / total_prob
        else:
            # 확률이 없으면 균등 분포
            for cluster in clusters:
                cluster['probability'] = 1.0 / len(clusters)
        
        # 엔트로피 계산
        entropy = -sum(
            c['probability'] * math.log(c['probability'] + 1e-10)
            for c in clusters
            if c['probability'] > 0
        )
        
        # 메인 action의 확률 찾기
        main_action_prob = 0.0
        if main_action:
            for cluster in clusters:
                if any(self._check_action_equivalence(main_action, a) for a in cluster['actions']):
                    main_action_prob = cluster['probability']
                    break
        
        # 클러스터 정보 정리
        cluster_info = []
        for i, cluster in enumerate(sorted(clusters, key=lambda x: x['probability'], reverse=True)):
            cluster_info.append({
                'cluster_id': i,
                'representative_action': cluster['actions'][0],
                'all_actions': list(set(cluster['actions'])),
                'probability': cluster['probability'],
                'count': len(cluster['actions'])
            })
        
        return {
            'action_entropy': entropy,
            'action_prob': main_action_prob if main_action_prob > 0 else cluster_info[0]['probability'] if cluster_info else 0.0,
            'clusters': cluster_info,
            'num_unique_actions': len(clusters)
        }
    
    def _calculate_uncertainty(
        self,
        response_text: str,
        action: str,
        logprobs_info: Optional[Dict]
    ) -> Dict:
        """
        action에 대한 불확실도 계산
        
        Returns:
            {
                'action_entropy': float,  # 엔트로피 (불확실도)
                'action_prob': float,     # 선택된 action의 확률
                'top_candidates': List[Dict],  # 상위 후보 action들
                'token_logprobs': List[Dict]   # action 토큰들의 logprob
            }
        """
        if logprobs_info is None:
            return {
                'action_entropy': None,
                'action_prob': None,
                'top_candidates': [],
                'token_logprobs': []
            }
        
        # action 부분의 토큰 찾기
        action_tokens = self._extract_action_tokens(response_text, action, logprobs_info)
        
        if not action_tokens:
            return {
                'action_entropy': None,
                'action_prob': None,
                'top_candidates': [],
                'token_logprobs': []
            }
        
        # action 후보들 추출
        candidates = self._extract_action_candidates(action_tokens, logprobs_info)
        
        # 확률 계산
        probs = {}
        total_logprob = 0.0
        
        for candidate, token_indices in candidates.items():
            logprob_sum = sum(
                logprobs_info['content'][idx].logprob 
                for idx in token_indices 
                if idx < len(logprobs_info['content'])
            )
            prob = math.exp(logprob_sum)
            probs[candidate] = prob
            if candidate == action:
                total_logprob = logprob_sum
        
        # 정규화
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        
        # 엔트로피 계산
        entropy = -sum(p * math.log(p + 1e-10) for p in probs.values() if p > 0)
        
        # 상위 후보 정렬
        top_candidates = sorted(
            [{'action': k, 'probability': v} for k, v in probs.items()],
            key=lambda x: x['probability'],
            reverse=True
        )[:10]
        
        # 토큰별 logprob 정보
        token_logprobs = []
        for idx in action_tokens:
            if idx < len(logprobs_info['content']):
                token_info = logprobs_info['content'][idx]
                token_logprobs.append({
                    'token': token_info.token,
                    'logprob': token_info.logprob,
                    'top_alternatives': [
                        {'token': alt.token, 'logprob': alt.logprob}
                        for alt in (token_info.top_logprobs or [])
                    ]
                })
        
        return {
            'action_entropy': entropy,
            'action_prob': probs.get(action, 0.0),
            'top_candidates': top_candidates,
            'token_logprobs': token_logprobs
        }
    
    def _extract_action_tokens(
        self,
        response_text: str,
        action: str,
        logprobs_info: Dict
    ) -> List[int]:
        """
        response_text에서 action 부분에 해당하는 토큰 인덱스 찾기
        """
        # JSON에서 action 필드 찾기
        try:
            # JSON 파싱
            json_match = re.search(r'\{[^}]*"action"[^}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                action_value = str(parsed.get('action', ''))
            else:
                action_value = action
        except:
            action_value = action
        
        # action 값이 토큰 시퀀스에서 어디에 있는지 찾기
        # 간단한 방법: action 문자열이 포함된 토큰들 찾기
        action_tokens = []
        accumulated_text = ""
        
        for idx, token_info in enumerate(logprobs_info['content']):
            token_text = token_info.token
            accumulated_text += token_text
            
            # action 값이 포함되어 있는지 확인
            if action_value.lower() in accumulated_text.lower() or accumulated_text.lower() in action_value.lower():
                action_tokens.append(idx)
            
            # 너무 길어지면 리셋
            if len(accumulated_text) > len(action_value) * 2:
                accumulated_text = ""
        
        # 더 정확한 방법: "action" 키워드 이후의 값 찾기
        if not action_tokens:
            found_action_key = False
            for idx, token_info in enumerate(logprobs_info['content']):
                token_text = token_info.token.lower()
                if '"action"' in token_text or "'action'" in token_text:
                    found_action_key = True
                    continue
                
                if found_action_key:
                    # 콜론이나 따옴표 건너뛰기
                    if token_text.strip() in [':', '"', "'", ' ']:
                        continue
                    # 값 부분 시작
                    if token_text.strip():
                        action_tokens.append(idx)
                        # 값이 끝날 때까지 (따옴표나 콤마)
                        if '"' in token_text or "'" in token_text or ',' in token_text:
                            break
        
        return action_tokens[:10]  # 최대 10개 토큰
    
    def _extract_action_candidates(
        self,
        action_tokens: List[int],
        logprobs_info: Dict
    ) -> Dict[str, List[int]]:
        """
        action 토큰 위치에서 가능한 후보 action들 추출
        
        중요: 모든 후보에 대해 동일한 토큰 인덱스를 사용하여 공정한 확률 비교
        """
        candidates = {}
        
        if not action_tokens:
            return candidates
        
        # 첫 번째 토큰 인덱스 (모든 후보는 이 위치의 토큰을 비교)
        first_token_idx = action_tokens[0]
        
        if first_token_idx >= len(logprobs_info['content']):
            return candidates
        
        first_token = logprobs_info['content'][first_token_idx]
        
        # 실제 action의 첫 번째 토큰 추가
        # 주의: 실제 action이 여러 토큰일 수 있지만, 공정한 비교를 위해 첫 번째 토큰만 사용
        actual_token_text = first_token.token.strip().strip('"').strip("'")
        if actual_token_text and actual_token_text not in [' ', ':', ',', '\n', '\\n']:
            # 모든 후보와 동일하게 첫 번째 토큰만 사용
            candidates[actual_token_text] = [first_token_idx]
        
        # top_logprobs에서 대안 후보 추출
        if first_token.top_logprobs:
            for alt in first_token.top_logprobs:
                candidate = alt.token.strip().strip('"').strip("'")
                # 유효한 후보만 추가 (공백, 구두점 제외)
                if candidate and candidate not in [' ', ':', ',', '\n', '\\n']:
                    # 모든 후보에 대해 동일한 토큰 인덱스 사용 (공정한 비교)
                    candidates[candidate] = [first_token_idx]
        
        return candidates
    
    def visualize_uncertainty(
        self,
        uncertainty_info: Dict,
        save_path: Optional[str] = None
    ):
        """
        불확실도 정보 시각화 (Semantic Entropy 방식)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 의미적 클러스터별 확률
        if uncertainty_info.get('clusters'):
            clusters = uncertainty_info['clusters']
            # 상위 10개만 표시
            clusters = sorted(clusters, key=lambda x: x['probability'], reverse=True)[:10]
            
            actions = [c['representative_action'] for c in clusters]
            probs = [c['probability'] for c in clusters]
            counts = [c['count'] for c in clusters]
            
            bars = axes[0].barh(actions, probs, color='skyblue')
            axes[0].set_xlabel('Probability')
            axes[0].set_title(f'Semantic Action Clusters\nEntropy: {uncertainty_info.get("action_entropy", 0):.4f}')
            axes[0].grid(axis='x', alpha=0.3)
            
            # 각 막대에 샘플 개수 표시
            for i, (bar, count) in enumerate(zip(bars, counts)):
                width = bar.get_width()
                axes[0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'n={count}', va='center', fontsize=8)
        else:
            axes[0].text(0.5, 0.5, 'No cluster data', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Action Clusters')
        
        # 2. 엔트로피 및 통계 정보
        axes[1].axis('off')
        info_text = f"""
Semantic Entropy Analysis

Entropy: {uncertainty_info.get('action_entropy', 'N/A'):.4f if uncertainty_info.get('action_entropy') is not None else 'N/A'}
Action Probability: {uncertainty_info.get('action_prob', 'N/A'):.4f if uncertainty_info.get('action_prob') is not None else 'N/A'}
Unique Actions: {uncertainty_info.get('num_unique_actions', 0)}

Clusters:
"""
        if uncertainty_info.get('clusters'):
            for i, cluster in enumerate(uncertainty_info['clusters'][:5]):
                info_text += f"\n{i+1}. {cluster['representative_action']}: {cluster['probability']:.4f} (n={cluster['count']})"
        
        axes[1].text(0.1, 0.9, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"시각화 저장: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_prompt_uncertainty(
        self,
        prompts: List[str],
        mission: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        여러 프롬프트에 대해 불확실도 분석
        
        Args:
            prompts: 테스트할 프롬프트 리스트
            mission: 미션 텍스트
        
        Returns:
            {
                'prompt': str,
                'entropy': float,
                'action_prob': float,
                'action': str
            } 리스트
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n프롬프트 {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                response = self.generate_action_with_uncertainty(
                    user_prompt=prompt,
                    mission=mission
                )
                
                uncertainty = response.get('uncertainty', {})
                
                results.append({
                    'prompt': prompt,
                    'entropy': uncertainty.get('action_entropy'),
                    'action_prob': uncertainty.get('action_prob'),
                    'action': response.get('action', ''),
                    'clusters': uncertainty.get('clusters', []),
                    'num_unique_actions': uncertainty.get('num_unique_actions', 0)
                })
                
                # 시각화 저장
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"uncertainty_prompt_{i+1}.png")
                    self.visualize_uncertainty(uncertainty, save_path)
                
            except Exception as e:
                print(f"오류 발생: {e}")
                results.append({
                    'prompt': prompt,
                    'entropy': None,
                    'action_prob': None,
                    'action': '',
                    'top_candidates': []
                })
        
        return results
    
    def visualize_prompt_comparison(
        self,
        analysis_results: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        여러 프롬프트에 대한 불확실도 비교 시각화
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 엔트로피 비교
        prompts_short = [r['prompt'][:30] + '...' if len(r['prompt']) > 30 else r['prompt'] 
                         for r in analysis_results]
        entropies = [r['entropy'] if r['entropy'] is not None else 0 for r in analysis_results]
        action_probs = [r['action_prob'] if r['action_prob'] is not None else 0 for r in analysis_results]
        
        x = range(len(prompts_short))
        
        axes[0].plot(x, entropies, marker='o', label='Entropy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(prompts_short, rotation=45, ha='right')
        axes[0].set_ylabel('Entropy (Uncertainty)')
        axes[0].set_title('Action Uncertainty by Prompt')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        # Action 확률 비교
        axes[1].plot(x, action_probs, marker='s', color='green', label='Action Probability')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(prompts_short, rotation=45, ha='right')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Selected Action Probability by Prompt')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"비교 시각화 저장: {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_scenario2_environment():
    """시나리오 2 환경 생성"""
    size = 10
    
    walls = []
    for i in range(size):
        walls.append((i, 0))
        walls.append((i, size-1))
        walls.append((0, i))
        walls.append((size-1, i))
    
    blue_pillar_positions = [(3, 4), (4, 4), (3, 5), (4, 5)]
    for pos in blue_pillar_positions:
        walls.append((pos[0], pos[1], 'blue'))
    
    table_positions = [(5, 1), (6, 1), (7, 1)]
    for pos in table_positions:
        walls.append((pos[0], pos[1], 'purple'))
    
    start_pos = (1, 8)
    goal_pos = (8, 1)
    
    room_config = {
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'walls': walls,
        'objects': []
    }
    
    # 절대 움직임 모드 활성화 (프롬프트에서 "up", "right" 등의 절대 방향 사용)
    return MiniGridEmojiWrapper(size=size, room_config=room_config, use_absolute_movement=True)


def main():
    """메인 함수: 불확실도 측정 예제"""
    print("=" * 60)
    print("VLM Action Uncertainty Estimation")
    print("=" * 60)
    
    # 환경 생성
    env = create_scenario2_environment()
    env.reset()
    
    # 불확실도 측정 컨트롤러 생성
    controller = UncertaintyController(
        env=env,
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        num_samples=5,  # Semantic Entropy 계산을 위한 샘플 수
        sampling_temperature=1.0  # 샘플링 다양성을 위한 temperature
    )
    
    # 테스트할 프롬프트들
    test_prompts = [
        "Go to the blue pillar",
        "Move towards the blue pillar",
        "Head to the blue pillar and turn right",
        "Navigate to the blue pillar located in the center",
        "Move up and right to reach the blue pillar"
    ]
    
    mission = "파란 기둥으로 가서 오른쪽으로 돌고, 테이블 옆에 멈추시오"
    
    # 불확실도 분석
    print("\n프롬프트별 불확실도 분석 시작...")
    results = controller.analyze_prompt_uncertainty(
        prompts=test_prompts,
        mission=mission,
        save_dir="uncertainty_plots"
    )
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("분석 결과")
    print("=" * 60)
    for i, result in enumerate(results):
        print(f"\n프롬프트 {i+1}: {result['prompt']}")
        print(f"  Action: {result['action']}")
        print(f"  Entropy: {result['entropy']:.4f}" if result['entropy'] else "  Entropy: N/A")
        print(f"  Action Probability: {result['action_prob']:.4f}" if result['action_prob'] else "  Action Probability: N/A")
        if result['top_candidates']:
            print(f"  Top 3 Candidates:")
            for j, cand in enumerate(result['top_candidates'][:3]):
                print(f"    {j+1}. {cand['action']}: {cand['probability']:.4f}")
    
    # 비교 시각화
    controller.visualize_prompt_comparison(
        results,
        save_path="uncertainty_plots/prompt_comparison.png"
    )
    
    print("\n분석 완료!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

