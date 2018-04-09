# Place this script in the subdirectory containing the simulation's results.
# The script expects a subdir called "results" that contains the csv files

library(data.table)
library(parallel)
num_cores<-detectCores(all.tests = FALSE, logical = FALSE)
print("---Executing in--")
print(num_cores)
print("-----------------")
root = "."
dir=paste(root,"/results",sep = "")
my_data_tot=data.table()
#find csv files
files=list.files(dir)
files=sapply(files,function(x) strsplit(x,".",fixed = TRUE)) #split name and extension
files=Filter(function(x) x[length(x)]=="csv",files) #remove all non-csv files
files=Filter(function(x) grepl("agent",x[1]),files) #remove all non-csv files
files=lapply(files,FUN=function(x) paste(unlist(x),collapse = "."))
files=sapply(unlist(files),function(x)paste(dir,x,sep="/"))
#read files in parallels
my_data_tot<-mclapply(files,FUN=fread,mc.cores=num_cores)
my_data_tot=rbindlist(my_data_tot)
my_data_tot=my_data_tot[,c(1,7:ncol(my_data_tot)),with=FALSE] #drop some columns
rm(files,dir)
#aggregate data
my_data_tot=my_data_tot[my_data_tot$seed!="-1",] # remove lines about food sources
## print avg behavior of agents
for(seed_lvl in levels(as.factor(my_data_tot$seed))) {
  stats<-mclapply(c("mean","sd"),function(f) {
    my_data_tot[seed==seed_lvl,lapply(.SD,eval(parse(text=f))),by=timeStep]
  },mc.cores=num_cores)
  eval(parse(text=paste("my_data_means_",seed_lvl,"=as.data.frame(stats[1])",sep="")))
  eval(parse(text=paste("my_data_sds_",seed_lvl,"=as.data.frame(stats[2])",sep="")))
  rm(stats)
  #write to file
  write.csv(get(paste("my_data_means_",seed_lvl,sep="")),paste(root,"/stats_means_",seed_lvl,".csv",sep = ""))
  write.csv(get(paste("my_data_sds_",seed_lvl,sep="")),paste(root,"/stats_sds_",seed_lvl,".csv",sep = ""))
  #compress csv files
  # mclapply(files,function(x)
  # {
  #   system(paste("bzip2 ",x,sep = ""))
  # },mc.cores=length(files))
  #plot
  n=ncol(get(paste("my_data_means_",seed_lvl,sep="")))-1
  colors = rainbow(n, s = 1, v = 1, start = 0, end = max(1, n - 1)/n, alpha = 1)
  rm(n)
  png(paste(root,"/results_",seed_lvl,".png",sep = ""),width=20,height = 20,units="cm",res = 300)
  x=get(paste("my_data_means_",seed_lvl,sep=""))$timeStep
  plot(x,xlim=c(min(x),max(x)),
       ##main=paste("Avg population performance, seed",seed_lvl),
      type="n",ylab="Action",xlab="Iterations",mgp=c(2.5,1,0),ylim=c(-0.5,4.5),yaxt="n",cex.lab=2,cex.axis=1.5)
  rm(x)
  actions=c("north","west","east","south","eat","")
  axis(2, at=0:6,labels=FALSE)
  abline(h=0,col = "gray60")
  abline(h=1,col = "gray60",lty=3)
  abline(h=2,col = "gray60",lty=3)
  abline(h=3,col = "gray60",lty=3)
  abline(h=4,col = "gray60",lty=3)
  abline(h=5,col = "gray60",lty=3)
  text(y=0:6+0.2, x=par()$usr[1]-0.05*(par()$usr[2]-par()$usr[1])
       ,labels=actions, srt=90, adj=1, xpd=TRUE)
  rm(actions)
  start_col=3
  legend("bottomright",names(get(paste("my_data_means_",seed_lvl,sep=""))[,-(1:start_col-1)]),fill=colors,horiz=TRUE)
  for(i in start_col:ncol(get(paste("my_data_means_",seed_lvl,sep="")))) {
    lines(get(paste("my_data_means_",seed_lvl,sep=""))[[1]],get(paste("my_data_means_",seed_lvl,sep=""))[[i]],col=colors[i-start_col+1],lwd=4)
    ## lines(get(paste("my_data_means_",seed_lvl,sep=""))[[1]],get(paste("my_data_means_",seed_lvl,sep=""))[[i]]+get(paste("my_data_sds_",seed_lvl,sep=""))[[i]],col=colors[i-start_col+1],lty=3,lwd=2)
    ## lines(get(paste("my_data_means_",seed_lvl,sep=""))[[1]],get(paste("my_data_means_",seed_lvl,sep=""))[[i]]-get(paste("my_data_sds_",seed_lvl,sep=""))[[i]],col=colors[i-start_col+1],lty=3,lwd=2)
  }
  rm(start_col)
  eval(parse(text=paste("rm(my_data_means_",seed_lvl,",my_data_sds_",seed_lvl,")",sep="")))
  dev.off()
}
library(plyr)
my_data_aggr=count(my_data_tot,c('seed','timeStep'))
write.csv(my_data_aggr,"population_evolution.csv")
#plot evolution of groups in population
n=length(levels(as.factor(my_data_aggr$seed)))
colors = c('Blue','Red') #rainbow(n, s = 1, v = 1, start = 0, end = max(1, n - 1)/n, alpha = 1)
#plot pdf
pdf(paste(root,"population_evolution.pdf",sep = "/"))
x=as.numeric(levels(as.factor(my_data_aggr$timeStep)))
y=as.numeric(levels(as.factor(my_data_aggr$freq)))
plot(x,xlim=c(min(x),max(x)),ylim=c(min(y)-200,max(y)),main="Groups sizes",type="n",ylab="Size",xlab="Time",cex.lab=2,cex.axis=1.5)
legend("bottomleft",c("Random","Sociable"),col=colors,lty=c(1,1),lwd=c(4,4),horiz=TRUE)
for(seed_lvl in levels(as.factor(my_data_aggr$seed))) {
  x=my_data_aggr[my_data_aggr$seed==seed_lvl,]$timeStep
  y=my_data_aggr[my_data_aggr$seed==seed_lvl,]$freq
  lines(x,y,col=colors[as.numeric(seed_lvl)+1],lwd=4)
  }
dev.off()
#plot png
png(paste(root,"population_evolution.png",sep = "/"),width=20,height = 20,units="cm",res = 300)
x=as.numeric(levels(as.factor(my_data_aggr$timeStep)))
y=as.numeric(levels(as.factor(my_data_aggr$freq)))
plot(x,xlim=c(min(x),max(x)),ylim=c(min(y)-200,max(y)),main="Groups sizes",type="n",ylab="Size",xlab="Time",cex.lab=2,cex.axis=1.5)
legend("bottomleft",c("Random","Sociable"),col=colors,lty=c(1,1),lwd=c(4,4),horiz=TRUE)
for(seed_lvl in levels(as.factor(my_data_aggr$seed))) {
  x=my_data_aggr[my_data_aggr$seed==seed_lvl,]$timeStep
  y=my_data_aggr[my_data_aggr$seed==seed_lvl,]$freq
  lines(x,y,col=colors[as.numeric(seed_lvl)+1],lwd=4)
}
dev.off()
rm(x,y,colors,n)
rm(my_data_tot,seed_lvl,root,my_data_aggr)
