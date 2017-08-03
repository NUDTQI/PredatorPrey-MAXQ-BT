#pragma once
#include "Action.h"
#include "Evaluator.h"

class Prey;

/*  zhangqi 2017/02/24
	���ܣ���������������ѧϰ�Ļ���ģ�ͻ��࣬��Ϊѧϰ�㷨ʹ�õĹ����ӿڣ�
	��ֻӦ��������ʵ�塢״̬��ָ�룬������ָ�룬�Լ����������ӿ�
*/

class EnvModelLearning
{
public:
	EnvModelLearning(Prey* pA,Evaluator* pEva)
	{
		m_pOwner = pA;
		pState = NULL;
		preState = NULL;
		pAction = NULL;
		preAction = NULL;
		rewardFeedback = 0;
		pMyEvaluator = pEva;
		durationCount = 0;
	}
	~EnvModelLearning(void){}
protected:
	Prey* m_pOwner;

public:
	//state
	CState* pState;
	CState* preState;
	//action
	CAction* pAction;
	CAction* preAction;
	//action data
	CActionData localActionData;
	CActionData preActionData;
	//reward
	Evaluator* pMyEvaluator;
	float rewardFeedback;
	//experineced state list
	std::vector<CState*> seq;
	std::vector<int> seqDur;
	int startime;
	int durationCount;

public:
	virtual float CalculateRewardsEachStep(){return pMyEvaluator->Evaluate(m_pOwner,preState,pState);}
	virtual CState* UpdateCurState(){return NULL;}
	virtual void SetCurAction(CAction* pA){pAction = pA;}
	virtual CState* MemyCopyState(CState* temp){return NULL;}
	virtual void bindActionData(CAction* pA, CActionData* pAD){pA->setActionData(pAD);}
};
