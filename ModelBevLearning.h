#pragma once
#include "Prey.h"
#include "AI/rl/EnvModel.h"
#include "AI/rl/Action.h"
#include "AI/rl/CPrimitiveAction.h"
#include "AI/rl/State.h"
#include "StateBevLearning.h"
#include "AI/RL/Evaluator.h"

//���ļ���Ҫ�û�Ϊ������Ҫѧϰ����Ϊ���ڵ㶨���佱������ģ�ͣ���Ӧ����Ƶ�״̬��
//���ڿ���ͳһ�Ľ���ѡ��������Ϊ�ӽڵ����ţ���˶���ģ������ʵ����


//����1��ʾretreat�ڵ㣻����2��ʾattack������3��ʾidle
class ModelSelRootBevLearning
	: public EnvModelLearning
{
public:

	ModelSelRootBevLearning(Prey* pA, Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelRootBevLearning(void){}

	StateRootBevSel temp;
	
public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateRootBevSel* ps = new StateRootBevSel();
		ps->CopyState(temp);
		return ps;
	}


	//��������Ҫ�����һ�����������״̬�����ռ��α����ֵ
	virtual float CalculateRewardsEachStep()
	{
		StateRootBevSel* pS = dynamic_cast<StateRootBevSel*>(pState);
		StateRootBevSel* preS = dynamic_cast<StateRootBevSel*>(preState);

		////////////////����ΪRoot�ڵ�ռ��ڶ���ĵ���α����ֵ����//////////////////////////////////////////////////////////
		float pseudoReward = 0;
		if (NonHl==pS->GetHealthlevel())
		{
			pseudoReward = pseudoReward - 10;
		}
		if (Inside==pS->GetDisToHaven())
		{
			pseudoReward = pseudoReward + 15;
		}

		//û�е��ˣ�û�з���havenѡ���˳ͷ�
		if (None==pS->GetDisToHaven() && None==pS->GetDisToEnemy() && preAction->getAction()==0)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//û�е��ˣ�ѡ������ͷ�
		if (None==pS->GetDisToEnemy() && preAction->getAction()==2)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//����˲�Զ��ѡ�����γͷ�
		if (Middle>pS->GetDisToEnemy() && preAction->getAction()==1)
		{
			pseudoReward = pseudoReward - 0.5;
		}

		//Ѫ�����ӣ�����
		if (pS->GetHealthlevel() > preS->GetHealthlevel())
		{
			pseudoReward = pseudoReward + 2;
		}

		if (LowHl == pS->GetHealthlevel())
		{
			if (Near==pS->GetDisToEnemy() && preAction->getAction()==0)
			{
				pseudoReward = pseudoReward + 0.5;
			}
		}

		//���Ѷ�Ѫ���߹�������
		if(HighHl==pS->GetHealthlevel() && LowAlly<pS->GetNumAllyNeighbor() && None>pS->GetDisToEnemy() && preAction->getAction()==2)
		{
			pseudoReward = pseudoReward + 0.5;
		}		
		return pseudoReward;
	}
};

class ModelSelRetreatBevLearning
	: public EnvModelLearning
{
public:

	ModelSelRetreatBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelRetreatBevLearning(void){}

	StateRetreatBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateRetreatBevSel* ps = new StateRetreatBevSel();
		ps->CopyState(temp);
		return ps;
	}

	virtual float CalculateRewardsEachStep()
	{
		StateRetreatBevSel* pS = dynamic_cast<StateRetreatBevSel*>(pState);

		////////////////����Ϊretreat�ڵ�ռ��ڶ���ĵ���α����ֵ����//////////////////////////////////////////////////////////
		float pseudoReward = 0;
		//���ڳ�����Ϊ�����������������������͵��ﰲȫ��������
		if (NonHl == pS->GetHealth())
		{
			pseudoReward = pseudoReward - 10;
		}

		if (Inside == pS->GetDisToHaven())
		{
			pseudoReward = pseudoReward + 15;
		}

		//û��Haven��ѡ�񵽴�Haven�ͷ�
		if (None==pS->GetDisToHaven() && preAction->getAction()==1)
		{
			pseudoReward = pseudoReward - 0.5;
		}
		return pseudoReward;
	}
};

class ModelSelAttackBevLearning
	: public EnvModelLearning
{
public:

	ModelSelAttackBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){}

	~ModelSelAttackBevLearning(void){}

	StateAttackBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateAttackBevSel* ps = new StateAttackBevSel();
		ps->CopyState(temp);
		return ps;
	}

	virtual float CalculateRewardsEachStep()
	{
		float pseudoReward = 0; 
		StateAttackBevSel* pS = dynamic_cast<StateAttackBevSel*>(pState);

		////////////////����Ϊattack�ڵ�ռ��ڶ���ĵ���α����ֵ����//////////////////////////////////////////////////////////
		//float pseudoReward = 0;
		//���ڽ�����Ϊ�����������������������͵��ﰲȫ��������

		if (NonHl == pS->GetHealthlevel())
		{
			pseudoReward = pseudoReward - 10;
		}

		if (m_pOwner->GetNearestPredator() && m_pOwner->GetNearestPredator()->isDead())
		{
			pseudoReward = pseudoReward + 2;
		}

		return pseudoReward;
	}
};


//general learning 
class ModelSelGeneralBevLearning
	: public EnvModelLearning
{
public:

	ModelSelGeneralBevLearning(Prey* pA,Evaluator* pEva)
		: EnvModelLearning(pA,pEva){startime = 0;}

	~ModelSelGeneralBevLearning(void){}

	StateGeneralBevSel temp;

public:

	virtual CState* UpdateCurState()
	{
		temp.UpdateDiscreteState(m_pOwner);
		return &temp;
	}

	virtual void SetCurAction(CAction* pA)
	{
		pAction->setAction(pA->getAction());
	}

	virtual CState* MemyCopyState(CState* temp)
	{
		StateGeneralBevSel* ps = new StateGeneralBevSel();
		ps->CopyState(temp);
		return ps;
	}
};