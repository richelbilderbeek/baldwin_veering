from PyPDF2 import PdfFileWriter, PdfFileReader
import os
import argparse

""" define parser: takes in three arguments, ranges = 2 args, file = 1 arg which determine the range and the output filename """
parser = argparse.ArgumentParser(description='Determine range of program and filename.')
parser.add_argument('--working_dir', metavar='working_dir', nargs=1,default="data",help='the data directory')
parser.add_argument('--treatment', metavar='treatment', nargs=1,default="0_special",help='the treatment directory')

args = parser.parse_args()

workingdir=args.working_dir[0]
t=args.treatment[0]

Dirs=["500","1000","10000"]

Names=["action_freq.pdf" , "play_freq.pdf" ,
        "skill_0_age_heat.pdf" , "skill_0_age.pdf",
        "skill_0_bins_heat.pdf" , "skill_0_bins.pdf",
        "skill_0_gen_age_heat.pdf" , "skill_0_gen_age.pdf",
        "skill_0_gen_bins_heat.pdf" , "skill_0_gen_bins.pdf",
        "env_agents_in_cell_agents_mean.pdf", "env_times_shared_shared.pdf",
        "env_agents_in_cell_agents_sum.pdf", "reprod.pdf","age_hist.pdf"]

num_cols=len(Dirs)
num_rows=len(Names)
col_size=200
row_size=200
totalSize=[col_size*num_cols,row_size*num_rows]

def addGraph(ix,iy,fname,basePage,col_size,row_size,totalSize):
        input1 = PdfFileReader(fname)
        page1 = input1.getPage(0)
        #position
        tx=col_size*ix
        ty=totalSize[1]-row_size*(iy+1)

        #scale
        sx=float(col_size)/float(page1.mediaBox[2])
        sy=float(row_size)/float(page1.mediaBox[3])

        if sx<sy:
                scale=sx
        else:
                scale=sy

        page1.scale(sx, sy)

        #MERGE
        basePage.mergeTranslatedPage(page1, tx, ty)

iy=0
output = PdfFileWriter()
output.addBlankPage(totalSize[0],totalSize[1])
basePage=output.getPage(0)
for file_name in Names:
        ix=0
        for simLen in Dirs:
                fname=os.path.join(workingdir,simLen,"vis",file_name)
                if os.path.exists(fname):
                        print(fname, ix, iy )
                        addGraph(ix,iy,fname,basePage,col_size,row_size,totalSize)
                ix+=1
        iy+=1

basePage.compressContentStreams()

# Output

outputStream = open(os.path.join(workingdir,"OUTPUT"+str(t)+".pdf"), "wb")
output.write(outputStream)
outputStream.close()
