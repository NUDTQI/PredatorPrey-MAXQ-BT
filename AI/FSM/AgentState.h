#ifndef CAGENTSTATE_H
#define CAGENTSTATE_H
#include <string>
/************************************************************************/
/*                          2012-3-6     ����                               
������״̬��������࣬��װ�˸�״̬��ִ�к���
*/
/************************************************************************/
#pragma once

enum EStatePriority   // ״̬�������ȼ�
{
	EStatePriority_Minimal = 0,
	EStatePriority_Minal = 1,
	EStatePriority_Midle = 2,
	EStatePriority_Upper = 3,
	EStatePriority_Most = 4
};

template <typename entity_type>
class CAgentState
{
public:
	entity_type *m_pOwner;
	virtual ~CAgentState(){};
	CAgentState()
	{
		m_pOwner = 0;
		m_ExecuteTime = 0;
		m_bIsDone = false;
		m_ParaInfo = L"none";
		m_CurPhase = 0;
		m_priority = EStatePriority_Midle;    // Ĭ��״̬�������ȼ����е�
	};
	virtual bool Enter(void)=0;       // ��һ�ν���״̬��Ҫ���������
	virtual void Execute(void)=0;     // ��״̬��ÿһ��������Ҫ���������
	virtual bool IsDone(void)=0;      // ��鱾״̬�Ƿ�����ɣ����˳���
	virtual bool Exit(void)=0;        // �˳���״̬ǰ��Ҫ���������

	int m_CurPhase;                   // ��ǰִ�н׶�
	int m_ExecuteTime;                // ��ǰִ��ʱ��
	bool m_bIsDone;                   // �Ƿ�ִ���굱ǰ״̬
    std::wstring m_StateName;         // ����״̬������
	std::wstring m_ParaInfo;          // ״̬�������Ĳ�����Ϣ����#�Ÿ���
	EStatePriority m_priority;        // ״̬�����ȼ�
};
#endif




