"use client";

import {
  Notebook,
  Key,
  Trash,
  Info,
  XSquare,
  LinkBreak,
  Link,
  Plug,
  Bird,
  Brain,
} from "@phosphor-icons/react";
import {
  SiBookstack,
  SiConfluence,
  SiGithub,
  SiGoogledrive,
  SiJira,
  SiNotion,
  SiSlack,
} from "react-icons/si";
import { FaFile, FaGlobe } from "react-icons/fa";
import Image from "next/image";

interface IconProps {
  size?: string;
  className?: string;
}

const defaultTailwindCSS = "text-blue-400 my-auto flex flex-shrink-0";

export const PlugIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Plug size={size} className={className} />;
};

export const NotebookIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Notebook size={size} className={className} />;
};

export const KeyIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Key size={size} className={className} />;
};

export const TrashIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Trash size={size} className={className} />;
};

export const LinkBreakIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <LinkBreak size={size} className={className} />;
};

export const LinkIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Link size={size} className={className} />;
};

export const XSquareIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <XSquare size={size} className={className} />;
};

export const GlobeIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <FaGlobe size={size} className={className} />;
};

export const FileIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <FaFile size={size} className={className} />;
};

export const SlackIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiSlack size={size} className={className} />;
};

export const GithubIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiGithub size={size} className={className} />;
};

export const GoogleDriveIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiGoogledrive size={size} className={className} />;
};

export const BookstackIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiBookstack size={size} className={className} />;
};

export const ConfluenceIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiConfluence size={size} className={className} />;
};

export const JiraIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiJira size={size} className={className} />;
};

export const SlabIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => (
  <div
    style={{ width: `${size}px`, height: `${size}px` }}
    className={`w-[${size}px] h-[${size}px] ` + className}
  >
    <Image src="/SlabLogoBlue.png" alt="Logo" width="96" height="96" />
  </div>
);

export const InfoIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Info size={size} className={className} />;
};

export const BrainIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <Brain size={size} className={className} />;
};

export const NotionIcon = ({
  size = "16",
  className = defaultTailwindCSS,
}: IconProps) => {
  return <SiNotion size={size} className={className} />;
};
