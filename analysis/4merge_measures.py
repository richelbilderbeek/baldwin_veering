from PyPDF2 import PdfFileWriter, PdfFileReader
import os
import argparse
from io import BytesIO,StringIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

""" define parser: takes in three arguments, ranges = 2 args, file = 1 arg which determine the range and the output filename """
parser = argparse.ArgumentParser(description='Determine range of program and filename.')
parser.add_argument('--working_dir', metavar='working_dir', nargs=1,default="data",help='the data directory')
parser.add_argument('--treatment', metavar='treatment', nargs=1,default="0_special",help='the treatment directory')

args = parser.parse_args()

workingdir=args.working_dir[0]
t=args.treatment[0]

Dirs=["100","200","500","3000"]

Names=["action_freq.pdf","measures_binned_effective.pdf","measures_binned_intention.pdf",
        "skill_0_bins_heat.pdf" , "skill_0_gen_bins_heat.pdf",
        "env_agents_in_cell_agents_sum.pdf","food_proportion.pdf"]

num_cols=len(Dirs)
num_rows=len(Names)
col_size=200
row_size=200
totalSize=[col_size*num_cols,row_size*num_rows]

def addGraph(ix,iy,fname,basePage,col_size,row_size,totalSize):
        input1 = PdfFileReader(fname)
        page1 = input1.getPage(0)
        #page1.compressContentStreams()

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

output = PdfFileWriter()
output.addBlankPage(totalSize[0],totalSize[1]+20)
basePage=output.getPage(0)
for ix in range(len(Dirs)):
        simLen=Dirs[ix]
        packet = BytesIO()
        # create a new PDF with Reportlab
        can = canvas.Canvas(packet, pagesize=letter)
        can.drawString(0, 0, str(simLen))
        can.save()

        #move to the beginning of the StringIO buffer
        packet.seek(0)
        title = PdfFileReader(packet)
        #position
        tx=col_size*ix+col_size/2.0
        ty=totalSize[1]+10
        basePage.mergeTranslatedPage(title.getPage(0), tx, ty)
        for iy in range(len(Names)):
                file_name=Names[iy]
                #fname=path_to_file+"/vis/"+file_name
                fname=os.path.join(workingdir,simLen,"vis",file_name)
                if os.path.exists(fname):
                        print(fname, iy, ix )
                        addGraph(ix,iy,fname,basePage,col_size,row_size,totalSize)
                else:
                        print(fname+" not found")

basePage.compressContentStreams()

# Output

outputStream = open(os.path.join(workingdir,"OUTPUT_measures"+str(t)+".pdf"), "wb")
output.write(outputStream)
outputStream.close()
