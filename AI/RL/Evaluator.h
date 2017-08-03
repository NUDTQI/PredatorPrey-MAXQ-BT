#pragma once
#include "../../Prey.h"
#include "../../Predator.h"
#include "EnvModel.h"
#include "Action.h"
#include "CPrimitiveAction.h"
#include "State.h"
#include "../../StateBevLearning.h"


class Evaluator
{
public:
	Evaluator(){}
	~Evaluator(){}

	virtual float Evaluate(Prey* pA,CState* preState,CState* ps){return 0;}
};

class GeneralEvaluator
	:public Evaluator
{
public:
	GeneralEvaluator(){}
	~GeneralEvaluator(){}

	virtual float Evaluate(Prey* m_pOwner, CState* preState, CState* ps)
	{
		StateGeneralBevSel* pS = dynamic_cast<StateGeneralBevSel*>(ps);
		StateGeneralBevSel* preS = dynamic_cast<StateGeneralBevSel*>(preState);
		if(!preS) preS = pS;

		////////////////以下为Root节点空间内定义的单步伪奖励值函数//////////////////////////////////////////////////////////
		float reward = 0.0f;
		if (HealthLevel::NonHl == pS->GetHealthlevel())
		{
			reward = reward - 50;
		}

		if (DisLevel::Inside == pS->GetDisToHaven())
		{
			reward = reward + 200;
		}

		//if (m_pOwner->m_CurEnemy && m_pOwner->m_CurEnemy->isDead())
		//{
		//	reward = reward + 2;
		//	m_pOwner->m_CurEnemy = NULL;
		//}
		
		//if (preS->GetHealthlevel() < pS->GetHealthlevel())
		//{
		//	reward = reward + 5;
		//}

		//if (preS->GetHealthlevel() > pS->GetHealthlevel())
		//{
		//	reward = reward - 5;
		//}

		////没有Haven，选择到达Haven惩罚
		if (DisLevel::None==preS->GetDisToHaven() && m_pOwner->curPrimaryAct=="SeekSafety")
		{
			reward = reward - 1;
		}

		////没有敌人，选择进攻惩罚
		if (DisLevel::None==preS->GetDisToEnemy() && (m_pOwner->curPrimaryAct=="Charge"||m_pOwner->curPrimaryAct=="Assist"))
		{
			reward = reward - 1;
		}

		////没有Food，选择Forge惩罚
		if (DisLevel::None==preS->GetDisToFood() && m_pOwner->curPrimaryAct=="Forage")
		{
			reward = reward - 1;
		}
		//
		//if (HealthLevel::LowHl >= pS->GetHealthlevel() && DisLevel::Inside==pS->GetDisToFood() && m_pOwner->curPrimaryAct=="Eat")
		//{
		//	reward = reward + 1;
		//}

		/*if (HealthLevel::LowHl == pS->GetHealthlevel())
		{
			if (DisLevel::Near==pS->GetDisToEnemy() && 
				DisLevel::Near==pS->GetDisToHaven() && m_pOwner->curPrimaryAct=="SeekSafety")
			{
				reward = reward + 15;
			}

			if (DisLevel::Inside==pS->GetDisToFood() && m_pOwner->curPrimaryAct=="Eat")
			{
				reward = reward + 10;
			}

			if (DisLevel::Inside<pS->GetDisToFood() && m_pOwner->curPrimaryAct=="Forage")
			{
				reward = reward + 5;
			}

			if (DisLevel::Near==pS->GetDisToEnemy() && DisLevel::None==pS->GetDisToHaven() && m_pOwner->curPrimaryAct=="Flee")
			{
				reward = reward + 1;
			}
		}

		if (HealthLevel::HighHl==pS->GetHealthlevel() && NumAllyLevel::HighAlly==pS->GetNumAllyNeighbor() && m_pOwner->curPrimaryAct=="Charge" )
		{
			reward = reward + 0.6;
		}

		if (HealthLevel::LowHl<pS->GetHealthlevel() && NumAllyLevel::MidAlly==pS->GetNumAllyNeighbor() && m_pOwner->curPrimaryAct=="Assist")
		{
			reward = reward + 0.5;
		}*/

		m_pOwner->m_dReward = m_pOwner->m_dReward + reward;
		return reward;
	}
};