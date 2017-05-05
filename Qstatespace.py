# Copyright 2017 GATECH ECE6254 KDS17 TEAM. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


MAXDEPTH = 11

class getstate(object):
    def __init__(self):
        self.states = {
                'C':{
                    'i':[i for i in range(1,MAXDEPTH)],
                    'f':[3,5,8],
                    'l':[1],
                    'd':[1],#[64,128,256,512],
                    'n':[8,4,1]
                    },
                'P':{
                    'i':[i for i in range(1,MAXDEPTH)],
                    'f,l':[(3,2),(2,2)],#[(8,3),(5,2),(3,2)],
                    'n':[8,4,1]
                    },
                'FC':{
                    'i':[i for i in range(1,MAXDEPTH)],
                    'd':[512,256,128],
                    'n':[i for i in range(1,3)]
                    },
                'relu':{
                    'i':[i for i in range(1,MAXDEPTH)],
                    'n':[1]
                    },
                'DO':{
                    'i':[i for i in range(1,MAXDEPTH)],
                    'n':[1]
                    },           
                'start':
                    {
                    's':['start']      
                        },
                'TS':{
                    's':['prevstate']
                    }
                        }
                
    
    def countstates(self):
        S = []
        count = 0
        for i in self.states['start']['s']:
            S.append(('Start',0))
            count+=1
        for i in self.states['C']['i']:
            for f in self.states['C']['f']:
                for l in self.states['C']['l']:
                    for d in self.states['C']['d']:
                        for n in self.states['C']['n']:
                            S.append(('C',i,f,l,d,n,count))
                            count+=1
        for i in self.states['P']['i']:
            for fl in self.states['P']['f,l']:
                for n in self.states['P']['n']:
                    S.append(('P',i,fl[0],fl[1],n,count))
                    count+=1
#        for i in self.states['FC']['i']:
#            for n in self.states['FC']['n']:
#                for d in self.states['FC']['d']:
#                            S.append(('FC',i,n,d,count))
#                            count+=1
        for i in self.states['DO']['i']:
            for n in self.states['DO']['n']:
                S.append(('DO',i,n,count))
                count+=1
        for i in self.states['relu']['i']:
            for n in self.states['relu']['n']:
                S.append(('relu',i,n,count))
                count+=1
        for s in self.states['TS']:
            S.append(('Terminate',count))
            count+=1
        s = len(S)
        return s,S
    
    def countactions(self):
        NUMSTATES,S = self.countstates()
        A = {}
        counter = 0
        for s in S:
            if s[0] !='Terminate':
                r = ()
                for sprime in S:
                    if (s[0]=='C' or s[0]=='P' or s[0]=='FC' or s[0]=='DO' or s[0]=='relu'):
                        if s[1]==MAXDEPTH-1:
                            r=r+(S[-1][-1],) #terminate state for max layers
                        elif sprime[1]==s[1]+1:
                            if s[0]=='FC':
                                if s[1]==2:
                                    r=r+(S[-1][-1],) #terminate state
                                elif s[2]>=sprime[2]:
                                    r=r+(sprime[-1],)
                            elif s[0]=='C' or s[0]=='DO' or s[0]=='relu':
                                r=r+(sprime[-1],)
                            elif s[0]=='P' and sprime[0]!='P':
                                if sprime[0]=='FC':
                                    if s[4]<8:
                                        r=r+(sprime[-1],)
                                else:
                                    r=r+(sprime[-1],)
                            elif sprime[0]=='DO':
                                print(s[0])
                    elif s[0]=='Start':
                        if sprime[1]==s[1]+1:
                            r=r+(sprime[-1],)
                
                newA = tuple(set(r))
                lists = ()
                for item in newA:
                    lists= lists+(item,counter)
                    counter+=1
                A[s[-1]] = newA
                
        return counter,A                    

#def main():
#    
sa = getstate()
NUMSTATES, S = sa.countstates()
NUMACTIONS,A = sa.countactions()
##print(NUMSTATES,NUMACTIONS)
## #563,23551
#
#if __name__ == '__main__':
#     main()
