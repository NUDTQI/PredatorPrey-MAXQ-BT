#ifndef CSTATE_MACHINE_H
#define CSTATE_MACHINE_H

#pragma once
#include "AgentState.h"
template <class entity_type>
class CStateMachine
{
public:
	entity_type *m_pOwner;                    // ��״̬��ʵ��������ָ�����
	CAgentState<entity_type> *m_pCurrentState;// ��ǰ״ָ̬������
	CAgentState<entity_type> *m_pGlobalState; // ȫ��״ָ̬������
	

	// ״̬�����캯��
	CStateMachine(entity_type* owner):m_pOwner(owner),m_pCurrentState(0)
	{
		m_pCurrentState=NULL;
		m_pGlobalState=NULL;
	}

	// ��������
	~CStateMachine()
	{
		if (m_pCurrentState!=0)
		{
			//m_pCurrentState->Exit();
			delete m_pCurrentState;
			m_pCurrentState=NULL;
		}
		if (m_pGlobalState!=0)
		{
			//m_pGlobalState->Exit();
			delete m_pGlobalState;
			m_pGlobalState=NULL;
		}
	}

	// ״̬ת��
	bool  ChangeState(CAgentState<entity_type>* pNewState)
	{
		bool b = false;
		if(m_pCurrentState!=0)
		{
			// ����ǰ״̬���Ƿǿգ���ôҪ���ȼ����״̬�������ȼ��Ƿ���ڵ�ǰ״̬����
			// ���ǣ���Ҫִ�е�ǰ״̬���˳�������
			if(pNewState->m_priority>=m_pCurrentState->m_priority)
			{
				std::wstring NewStateName = pNewState->m_StateName,OldStateName = m_pCurrentState->m_StateName;
				if(NewStateName == OldStateName) 
				{
					b = false;
				}
				else 
				{
					b = m_pCurrentState->Exit();
				}
			}
		}  
		else b=true;      // ����ǰ״̬���ǿյģ���ֱ�ӿ��Խ����µ�״̬��
		if(!b){return b;}                                     // ����ǰ״̬�����˳�����״̬�л�ʧ��,��������
		b = pNewState->Enter();                               // ִ����״̬����ڲ���������Ϊ״̬�ĸ���ִ����ǰ��׼��
		if(!b){return b;}                                     // ����������״̬�������������״̬�л�ʧ�ܣ���������
		if (m_pCurrentState!=0){delete m_pCurrentState;}      // ����ǰ״̬�ǿգ�ɾ����ǰ״̬
		m_pCurrentState = pNewState;                          // ��״̬��ڲ����ɹ���ɣ����л��µ�״̬
		return b;
	}

	// ״̬ת��
	bool  ChangeGlobalState(CAgentState<entity_type>* pNewState)
	{
		bool b = true;
		if(m_pGlobalState!=0){b=m_pGlobalState->Exit();}      // ִ�е�ǰ״̬���˳�������
		if(!b){return b;}                                     // ����ǰ״̬�����˳�����״̬�л�ʧ��,��������
		b = pNewState->Enter();                               // ִ����״̬����ڲ���������Ϊ״̬�ĸ���ִ����ǰ��׼��
		if(!b){return b;}                                     // ����������״̬�������������״̬�л�ʧ�ܣ���������
		if (m_pGlobalState!=0){delete m_pGlobalState;}        // ����ǰ״̬�ǿգ�ɾ����ǰ״̬
		m_pGlobalState = pNewState;                           // ��״̬��ڲ����ɹ���ɣ����л��µ�״̬
		return b;
	}


	// ��ǰ״̬���£�����ǰ״̬��֮��Ϊ��ϣ�ɾ��֮������true�����򷵻�false;
	bool  Update()//const
	{
		// ִ�е�ǰ״̬
		bool b = false;
		if (m_pCurrentState) 
		{
			m_pCurrentState->Execute();
		    if(m_pCurrentState->IsDone())
	        {
			   m_pCurrentState->Exit();
	           delete m_pCurrentState;
	           m_pCurrentState=NULL;
	        }
		}
		else
		{
			// ��ǰ״̬Ϊ�գ�����ִ��
			b=true;
		}
		return b;
	}

	// ȫ��״̬����
	void UpdateGlobalState()
	{
		if (m_pGlobalState) 
		{
			m_pGlobalState->Execute();
		}
	}

    // ��ȫ�ر�״̬��
    void ShutdownStateMachine()
   {
       if(m_pGlobalState) {delete m_pGlobalState;m_pGlobalState=NULL;}
       if(m_pCurrentState){delete m_pCurrentState;m_pCurrentState=NULL;}
   }
};



#endif


